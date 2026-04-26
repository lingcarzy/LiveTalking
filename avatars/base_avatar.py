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
#
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
from pathlib import Path
import time
import cv2
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)

class BaseAvatar:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms)
        self.sessionid = self.opt.sessionid

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        # self.custom_opt = {}
        self.__loadcustom()

        self.batch_size = opt.batch_size
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()
        self._perf_window_start = time.perf_counter()
        self._perf = {
            'infer_batches': 0,
            'infer_frames': 0,
            'infer_busy_sec': 0.0,
            'process_frames': 0,
            'process_audio_chunks': 0,
            'process_busy_sec': 0.0,
        }
        self._watermark_text = str(getattr(opt, 'watermark_text', 'LiveTalking') or '').strip()
        self._watermark_enabled = bool(self._watermark_text)
        self._render_backpressure_threshold = int(getattr(opt, 'render_backpressure_threshold', 8))
        self._render_backpressure_streak_required = int(getattr(opt, 'render_backpressure_streak', 3))
        self._render_last_res_q = 0
        self._render_last_out_q = 0
        self._render_pressure_streak = 0
        self._watermark_cache = {}

        _tts_modules = {
            'edgetts': 'tts.edge',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'cosyvoice3': 'tts.cosyvoice3',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts'
        }

        if opt.tts in _tts_modules:
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict={}):
        if hasattr(self, 'tts'):
            self.tts.put_msg_txt(msg, datainfo)
    
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={}): # 16khz 20ms pcm
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo)

    def put_audio_file(self, filebyte, datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

    def put_audio_filepath(self, filepath, datainfo:dict={}): 
        stream = self.__create_bytes_stream(filepath)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        self.custom_audiotype = 0  

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            logger.info("notify:%s", eventpoint)

    def start_recording(self):
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        session_tag = str(self.opt.sessionid)
        audio_path = Path(f"temp{session_tag}.aac")
        video_path = Path(f"temp{session_tag}.mp4")
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"record_{session_tag}.mp4"
        command = [
            'ffmpeg',
            '-y',
            '-i',
            str(audio_path),
            '-i',
            str(video_path),
            '-c:v',
            'copy',
            '-c:a',
            'copy',
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            for temp_path in (audio_path, video_path):
                if temp_path.exists():
                    temp_path.unlink()

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1
        
    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            for _ in range(self.batch_size * 2):
                audioframe:AudioFrameData = self.asr.output_queue.get()
                if audioframe.type == 0:
                    is_all_silence = False               
                audio_frames.append(audioframe)

             # 检测状态变化
            current_speaking = not is_all_silence

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                    index = index + 1
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                pred = self.inference_batch(index, audiofeat_batch)

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], mirror_index(length, index)))
                    index = index + 1

            self._perf['infer_batches'] += 1
            self._perf['infer_frames'] += self.batch_size
            self._perf['infer_busy_sec'] += (time.perf_counter() - starttime)
            self._maybe_log_pipeline_stats()
                    
            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    def process_frames(self,quit_event):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        _last_speaking = False
        _transition_start = time.time()
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            frame_start = time.perf_counter()
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            if self._watermark_enabled and self._watermark_text:
                self._apply_watermark(combine_frame)
            
            # 使用统一输出接口推送视频帧
            self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                #frame,type,eventpoint = audio_frame
                frame = (audio_frame.data * 32767).astype(np.int16)

                # 使用统一输出接口推送音频帧
                self.output.push_audio_frame(frame, audio_frame.userdata)
                self.record_audio_data(frame)

            self._perf['process_frames'] += 1
            self._perf['process_audio_chunks'] += len(audio_frames)
            self._perf['process_busy_sec'] += (time.perf_counter() - frame_start)
            self._maybe_log_pipeline_stats()
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    def _maybe_log_pipeline_stats(self):
        now = time.perf_counter()
        window = now - self._perf_window_start
        if window < 5.0:
            return

        infer_fps = self._perf['infer_frames'] / max(1e-6, window)
        process_fps = self._perf['process_frames'] / max(1e-6, window)
        audio_chunk_rate = self._perf['process_audio_chunks'] / max(1e-6, window)
        infer_busy_pct = (self._perf['infer_busy_sec'] / max(1e-6, window)) * 100.0
        process_busy_pct = (self._perf['process_busy_sec'] / max(1e-6, window)) * 100.0

        asr_q = self.asr.queue.qsize() if hasattr(self, 'asr') and hasattr(self.asr, 'queue') else -1
        asr_out_q = self.asr.output_queue.qsize() if hasattr(self, 'asr') and hasattr(self.asr, 'output_queue') else -1
        asr_feat_q = self.asr.feat_queue.qsize() if hasattr(self, 'asr') and hasattr(self.asr, 'feat_queue') else -1
        res_q = self.res_frame_queue.qsize() if hasattr(self, 'res_frame_queue') else -1
        out_buf = self.output.get_buffer_size() if hasattr(self, 'output') else -1

        logger.info(
            'pipeline stats: infer_fps=%.2f process_fps=%.2f audio_chunk_rate=%.2f infer_busy=%.1f%% process_busy=%.1f%% q_asr=%d q_asr_out=%d q_feat=%d q_res=%d out_buf=%d speaking=%s',
            infer_fps,
            process_fps,
            audio_chunk_rate,
            infer_busy_pct,
            process_busy_pct,
            asr_q,
            asr_out_q,
            asr_feat_q,
            res_q,
            out_buf,
            self.speaking,
        )

        self._perf_window_start = now
        self._perf['infer_batches'] = 0
        self._perf['infer_frames'] = 0
        self._perf['infer_busy_sec'] = 0.0
        self._perf['process_frames'] = 0
        self._perf['process_audio_chunks'] = 0
        self._perf['process_busy_sec'] = 0.0

    def _apply_watermark(self, frame: NDArray[np.uint8]) -> None:
        h, w = frame.shape[:2]
        key = (h, w, self._watermark_text)
        cached = self._watermark_cache.get(key)
        if cached is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(self._watermark_text, font, font_scale, thickness)
            patch_h = max(1, th + baseline + 2)
            patch_w = max(1, tw + 2)
            patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
            cv2.putText(patch, self._watermark_text, (1, th + 1), font, font_scale, (128, 128, 128), thickness)
            mask = np.any(patch != 0, axis=2)
            cached = {
                'patch': patch,
                'mask': mask,
                'text_h': th,
            }
            self._watermark_cache[key] = cached

        x = 10
        y_baseline = 20
        patch = cached['patch']
        mask = cached['mask']
        top = max(0, y_baseline - cached['text_h'] - 1)
        left = max(0, x)
        bottom = min(h, top + patch.shape[0])
        right = min(w, left + patch.shape[1])
        if bottom <= top or right <= left:
            return

        patch_view = patch[: bottom - top, : right - left]
        mask_view = mask[: bottom - top, : right - left]
        roi = frame[top:bottom, left:right]
        roi[mask_view] = patch_view[mask_view]

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        while not quit_event.is_set(): 
            self.asr.run_step()

            res_q = self.res_frame_queue.qsize() if hasattr(self, 'res_frame_queue') else 0
            out_q = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0

            # Trigger stronger backpressure only when queue growth is sustained.
            if (
                res_q >= self._render_backpressure_threshold
                and out_q >= self._render_backpressure_threshold // 2
                and res_q >= self._render_last_res_q
                and out_q >= self._render_last_out_q
            ):
                self._render_pressure_streak += 1
            else:
                self._render_pressure_streak = 0

            self._render_last_res_q = res_q
            self._render_last_out_q = out_q

            if self._render_pressure_streak >= self._render_backpressure_streak_required:
                sleep_s = min(0.02, 0.0008 * (res_q + out_q))
                logger.debug('adaptive sleep res_q=%d out_q=%d streak=%d sleep=%.4f', res_q, out_q, self._render_pressure_streak, sleep_s)
                time.sleep(sleep_s)
            elif res_q > 0 or out_q > 0:
                time.sleep(0.0005)
            else:
                time.sleep(0.0002)
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()

    def shutdown(self):
        """Best-effort resource cleanup for session lifecycle management."""
        try:
            self.flush_talk()
        except Exception:
            logger.exception('flush_talk during shutdown failed')

        if getattr(self, 'quit_event', None) is not None:
            self.quit_event.set()

        if self.recording:
            try:
                self.stop_recording()
            except Exception:
                logger.exception('stop_recording during shutdown failed')

        if hasattr(self, 'output') and self.output is not None:
            try:
                self.output.stop()
            except Exception:
                logger.exception('output stop during shutdown failed')

