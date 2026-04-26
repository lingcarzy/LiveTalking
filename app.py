###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
import registry
from server.routes import setup_routes, LLM_EXECUTOR
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import asyncio
import torch
from pathlib import Path
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc


_NVENC_PATCH_APPLIED = False


app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None


def resolve_avatar_id(requested_avatar_id: str) -> str:
    avatar_id = str(requested_avatar_id or opt.avatar_id)
    avatar_root = Path("./data/avatars").resolve()
    candidate = (avatar_root / avatar_id).resolve()
    if not str(candidate).startswith(str(avatar_root)):
        raise ValueError("invalid avatar path")
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"avatar not found: {avatar_id}")
    return candidate.name

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = resolve_avatar_id(params.get('avatar', opt.avatar_id))
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        if len(custom_config) > opt.max_custom_config_chars:
            raise ValueError("custom_config too large")
        try:
            opt_this.customopt = json.loads(custom_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid custom_config json: {e}")

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session


def configure_webrtc_h264_encoder(opt) -> None:
    """Prefer NVENC for aiortc H264 encoding and fallback to software when unavailable."""
    global _NVENC_PATCH_APPLIED

    if _NVENC_PATCH_APPLIED:
        return

    if getattr(opt, 'transport', '') not in ('webrtc', 'rtcpush'):
        return

    encoder_mode = getattr(opt, 'webrtc_video_encoder', 'auto')
    if encoder_mode == 'software':
        logger.info('WebRTC encoder mode: software (libx264)')
        return

    try:
        import av
        import fractions
        from aiortc.codecs import h264 as aiortc_h264
    except Exception as e:
        logger.warning(f'WebRTC NVENC setup skipped: {e}')
        return

    warned_fallback = {'done': False}
    preset = getattr(opt, 'webrtc_nvenc_preset', 'p4')
    target_fps = int(getattr(opt, 'fps', 25) or 25)
    if target_fps <= 0:
        target_fps = 25

    def _create_nvenc_codec(width: int, height: int, bitrate: int):
        codec = av.CodecContext.create('h264_nvenc', 'w')
        codec.width = int(width)
        codec.height = int(height)
        codec.bit_rate = int(bitrate)
        codec.pix_fmt = 'yuv420p'
        codec.framerate = fractions.Fraction(target_fps, 1)
        codec.time_base = fractions.Fraction(1, target_fps)
        codec.options = {
            'preset': preset,
            'tune': 'll',
            'rc': 'cbr',
            'zerolatency': '1',
            'bf': '0',
        }
        return codec

    # aiortc>=1.14 uses H264Encoder._encode_frame directly.
    encoder_cls = getattr(aiortc_h264, 'H264Encoder', None)
    if encoder_cls is not None and hasattr(encoder_cls, '_encode_frame'):
        original_encode_frame = encoder_cls._encode_frame

        def _patched_encode_frame(self, frame, force_keyframe):
            # NVENC H264 expects yuv420p; ensure even dimensions for chroma subsampling.
            aligned_width = max(2, int(frame.width) & ~1)
            aligned_height = max(2, int(frame.height) & ~1)
            if frame.format.name != 'yuv420p' or aligned_width != frame.width or aligned_height != frame.height:
                frame = frame.reformat(width=aligned_width, height=aligned_height, format='yuv420p')

            if self.codec and (
                frame.width != self.codec.width
                or frame.height != self.codec.height
                or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
            ):
                self.buffer_data = b''
                self.buffer_pts = None
                self.codec = None

            if force_keyframe:
                frame.pict_type = av.video.frame.PictureType.I
            else:
                frame.pict_type = av.video.frame.PictureType.NONE

            if self.codec is None:
                try:
                    self.codec = _create_nvenc_codec(frame.width, frame.height, self.target_bitrate)
                except Exception as e:
                    if encoder_mode == 'nvenc':
                        raise
                    if not warned_fallback['done']:
                        warned_fallback['done'] = True
                        logger.warning(f'WebRTC NVENC unavailable, fallback to software encoder: {e}')

            if self.codec is None:
                return original_encode_frame(self, frame, force_keyframe)

            data_to_send = b''
            for package in self.codec.encode(frame):
                data_to_send += bytes(package)

            if data_to_send:
                yield from self._split_bitstream(data_to_send)

        encoder_cls._encode_frame = _patched_encode_frame
        _NVENC_PATCH_APPLIED = True
        logger.info('WebRTC encoder patch active: aiortc H264Encoder uses NVENC first, software fallback enabled')
        return

    # Backward compatibility for older aiortc versions.
    original_create = getattr(aiortc_h264, 'create_encoder_context', None)
    if callable(original_create):
        def _extract_by_index(args, kwargs, name, index):
            if name in kwargs:
                return kwargs[name]
            if len(args) > index:
                return args[index]
            return None

        def _patched_create_encoder_context(codec_name, *args, **kwargs):
            if codec_name == 'libx264':
                width = _extract_by_index(args, kwargs, 'width', 0)
                height = _extract_by_index(args, kwargs, 'height', 1)
                bitrate = _extract_by_index(args, kwargs, 'bitrate', 2)
                try:
                    return _create_nvenc_codec(width, height, bitrate)
                except Exception as e:
                    if encoder_mode == 'nvenc':
                        raise
                    if not warned_fallback['done']:
                        warned_fallback['done'] = True
                        logger.warning(f'WebRTC NVENC unavailable, fallback to software encoder: {e}')
            return original_create(codec_name, *args, **kwargs)

        aiortc_h264.create_encoder_context = _patched_create_encoder_context
        _NVENC_PATCH_APPLIED = True
        logger.info('WebRTC encoder patch active: create_encoder_context hooks NVENC first, software fallback enabled')
        return

    logger.warning('WebRTC NVENC setup skipped: unsupported aiortc h264 encoder internals')


def run_webrtc_encoder_selfcheck(opt) -> None:
    """Log startup encoder capability summary to quickly diagnose finalfps bottlenecks."""
    if getattr(opt, 'transport', '') not in ('webrtc', 'rtcpush'):
        return

    encoder_mode = getattr(opt, 'webrtc_video_encoder', 'auto')
    preset = getattr(opt, 'webrtc_nvenc_preset', 'p4')

    try:
        import av
    except Exception as e:
        logger.warning(f'WebRTC encoder self-check skipped: cannot import PyAV: {e}')
        return

    libx264_ok = False
    libx264_err = ''
    try:
        av.CodecContext.create('libx264', 'w')
        libx264_ok = True
    except Exception as e:
        libx264_err = str(e)

    nvenc_ok = False
    nvenc_err = ''
    try:
        av.CodecContext.create('h264_nvenc', 'w')
        nvenc_ok = True
    except Exception as e:
        nvenc_err = str(e)

    logger.info(
        'WebRTC encoder self-check: mode=%s preset=%s nvenc=%s libx264=%s',
        encoder_mode,
        preset,
        'ok' if nvenc_ok else 'unavailable',
        'ok' if libx264_ok else 'unavailable',
    )

    if not nvenc_ok and encoder_mode in ('auto', 'nvenc'):
        logger.warning('WebRTC NVENC self-check detail: %s', nvenc_err)
    if not libx264_ok:
        logger.warning('WebRTC software encoder self-check detail: %s', libx264_err)

    if encoder_mode == 'nvenc' and not nvenc_ok:
        logger.warning('WEBRTC_VIDEO_ENCODER=nvenc but NVENC is unavailable; video encode may fail')
    elif encoder_mode == 'auto' and not nvenc_ok and libx264_ok:
        logger.info('WebRTC will use software encoder fallback (libx264)')

async def offer(request):
    return await rtc_manager.handle_offer(request)

async def on_shutdown(app):
    await rtc_manager.shutdown()
    LLM_EXECUTOR.shutdown(wait=False, cancel_futures=True)



def main():
    global rtc_manager, opt, model,load_avatar
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()
    logger.info(
        'runtime tuning: batch_size=%d asr_feat_queue_size=%d fps=%d transport=%s encoder=%s',
        int(getattr(opt, 'batch_size', 0)),
        int(getattr(opt, 'asr_feat_queue_size', 0)),
        int(getattr(opt, 'fps', 0)),
        str(getattr(opt, 'transport', '')),
        str(getattr(opt, 'webrtc_video_encoder', '')),
    )
    configure_webrtc_h264_encoder(opt)
    run_webrtc_encoder_selfcheck(opt)

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    if opt.model == 'musetalk':
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,global_avatars[opt.avatar_id],160)

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    # share avatar_sessions (RTCManager handles it but routes.py expects it)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["max_chat_chars"] = opt.max_chat_chars
    appasync["max_audio_upload_bytes"] = opt.max_audio_upload_mb * 1024 * 1024

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    
    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync) 

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
    
    
    
