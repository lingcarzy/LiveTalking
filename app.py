# app.py
import sys
import os
import signal
import time

# 将当前文件所在的目录（项目根目录）添加到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import gc
import json
import random
import ipaddress
import torch.multiprocessing as mp
from typing import Dict, Optional

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

# 引入新的配置模块
from configs import AppConfig
from core.session_manager import session_manager
from core.render_loop import RenderLoop
from core.muse_render import MuseRenderLoop
from core.lip_render import LipRenderLoop
from core.light_render import LightRenderLoop
from core.utils import (
    load_musetalk_model, load_musetalk_avatar, warm_up_musetalk,
    load_wav2lip_model, load_wav2lip_avatar, warm_up_wav2lip,
    load_ultralight_model, load_ultralight_avatar, warm_up_ultralight
)
from services.llm import create_llm_service
from webrtc import HumanPlayer
from logger import logger

# 全局变量
pcs = set()  # WebRTC 连接集合
config: Optional[AppConfig] = None
llm_service = None
model_instance = None
avatar_instance = None

# 新增：虚拟摄像头模式的全局退出控制
virtualcam_quit_event = None
virtualcam_render_thread = None

# =============================================================================
# 辅助函数
# =============================================================================
async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.error(f'HTTP POST Error: {e}')
        return None

async def run(push_url, sessionid):
    """P2: RTCPush 模式专用启动逻辑"""
    config.session_id = sessionid
    
    if config.model.name == 'musetalk':
         render_cls = MuseRenderLoop
    elif config.model.name == 'wav2lip':
         render_cls = LipRenderLoop
    elif config.model.name == 'ultralight':
         render_cls = LightRenderLoop
    else: return

    render_loop = render_cls(config, model_instance, avatar_instance)
    await session_manager.register_session(sessionid, render_loop)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Push Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            await session_manager.destroy_session(sessionid)

    player = HumanPlayer(render_loop, asyncio.get_running_loop())
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    
    if answer:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))
    else:
        logger.error(f"Failed to get answer from {push_url}")

# =============================================================================
# 路由处理函数 (保持不变)
# =============================================================================
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    try:
        nerfreal = await session_manager.create_session(config, model_instance, avatar_instance)

        sessionid = nerfreal.session_id
    except Exception as e:
        logger.exception("Failed to create session")
        return web.json_response({"code": -1, "msg": str(e)})

    logger.info(f'New session created: {sessionid}, total sessions: {len(session_manager._sessions)}')

    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Session {sessionid} - Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            await session_manager.destroy_session(sessionid)

    player = HumanPlayer(nerfreal, asyncio.get_running_loop())
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )

async def human(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        
        session = session_manager.get_session(sessionid)
        if not session:
            return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        if params.get('interrupt'):
            session.flush_talk()
        
        if params['type'] == 'echo':
            session.put_msg_txt(params['text'])
        elif params['type'] == 'chat':
            async def process_chat():
                try:
                    async for text_segment in llm_service.chat_stream(params['text']):
                        session.put_msg_txt(text_segment)
                except Exception as e:
                    logger.error(f"LLM chat stream error: {e}")
            
            def run_async_chat():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(process_chat())
                finally:
                    loop.close()

            if llm_service:
                asyncio.get_running_loop().run_in_executor(None, run_async_chat)
            else:
                logger.error("LLM service not initialized.")
                return web.json_response({"code": -1, "msg": "LLM service not ready"})
                
        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception("Error in /human")
        return web.json_response({"code": -1, "msg": str(e)})

async def interrupt_talk(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        
        session = session_manager.get_session(sessionid)
        if session:
            session.flush_talk()
            
        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception("Error in /interrupt_talk")
        return web.json_response({"code": -1, "msg": str(e)})

async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        
        session = session_manager.get_session(sessionid)
        if session:
            session.put_audio_file(filebytes)
        else:
             return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception("Error in /humanaudio")
        return web.json_response({"code": -1, "msg": str(e)})

async def set_audiotype(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        
        session = session_manager.get_session(sessionid)
        if session:
            session.set_custom_state(params['audiotype'], params.get('reinit', True))
        else:
             return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception("Error in /set_audiotype")
        return web.json_response({"code": -1, "msg": str(e)})

async def record(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        
        session = session_manager.get_session(sessionid)
        if not session:
             return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        if params['type'] == 'start_record':
            if len(session.frame_list_cycle) > 0:
                h, w, _ = session.frame_list_cycle[0].shape
                session.recorder.set_resolution(w, h)
            session.recorder.start()
        elif params['type'] == 'end_record':
            session.recorder.stop()
            
        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception("Error in /record")
        return web.json_response({"code": -1, "msg": str(e)})

async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    session = session_manager.get_session(sessionid)
    speaking = session.is_speaking() if session else False
    return web.json_response({"code": 0, "data": speaking})

async def on_shutdown(app):
    logger.info("Shutting down server...")
    # 关闭所有WebRTC连接
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros)
    pcs.clear()
    
    # 销毁所有会话
    for session_id in list(session_manager._sessions.keys()):
        await session_manager.destroy_session(session_id)

# =============================================================================
# 启动逻辑
# =============================================================================
def init_model():
    """P1: 使用新的加载路径"""
    global model_instance, avatar_instance, config
    
    logger.info(f"Initializing model: {config.model.name}")
    if config.model.name == 'musetalk':
        model_instance = load_musetalk_model()
        avatar_instance = load_musetalk_avatar(config.model.avatar_id)
        warm_up_musetalk(config.model.batch_size, model_instance)
        
    elif config.model.name == 'wav2lip':
        model_instance = load_wav2lip_model("./models/wav2lip.pth")
        avatar_instance = load_wav2lip_avatar(config.model.avatar_id)
        warm_up_wav2lip(config.model.batch_size, model_instance, 256)
        
    elif config.model.name == 'ultralight':
        audio_processor = load_ultralight_model()
        ultralight_model, frames, faces, coords = load_ultralight_avatar(config.model.avatar_id)
        model_instance = audio_processor
        avatar_instance = (ultralight_model, frames, faces, coords)
        warm_up_ultralight(config.model.batch_size, avatar_instance, 160)

async def async_shutdown(runner):
    """异步清理函数"""
    global virtualcam_quit_event, virtualcam_render_thread
    
    logger.info("Starting async shutdown...")
    
    # 如果是虚拟摄像头模式，停止渲染线程
    if virtualcam_quit_event is not None:
        virtualcam_quit_event.set()
        if virtualcam_render_thread is not None:
            virtualcam_render_thread.join(timeout=5)
    
    # 清理WebRTC连接
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    
    # 销毁所有会话
    for session_id in list(session_manager._sessions.keys()):
        try:
            await session_manager.destroy_session(session_id)
        except Exception as e:
            logger.error(f"Error destroying session {session_id}: {e}")
    
    # 清理runner
    await runner.cleanup()

def run_server(runner):
    """修改后的服务器启动函数，正确处理信号和清理"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', config.server.listen_port)
        loop.run_until_complete(site.start())
        
        # 如果是 rtcpush 模式，启动推流
        if config.server.transport == 'rtcpush':
            for k in range(config.server.max_session):
                push_url = config.server.push_url
                if k != 0:
                    push_url = config.server.push_url + str(k)
                loop.run_until_complete(run(push_url, k))
        
        # 信号处理函数
        def signal_handler():
            logger.info("Received shutdown signal")
            asyncio.ensure_future(async_shutdown(runner))
            loop.stop()
        
        # 注册信号处理器（Windows下可能不支持）
        try:
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        except NotImplementedError:
            # Windows系统下不支持add_signal_handler，使用其他方法
            pass
        
        logger.info("Server started successfully, press Ctrl+C to stop")
        
        # 运行事件循环
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Cleaning up...")
        try:
            # 执行清理
            loop.run_until_complete(async_shutdown(runner))
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            loop.stop()
            loop.close()
            logger.info("Server stopped")
# =============================================================================
# 自定义 CORS 中间件：智能放行局域网，拒绝公网
# =============================================================================
def is_local_origin(origin: str) -> bool:
    """判断请求来源是否属于内网或本地"""
    if not origin:
        return False
        
    origin_host = origin.split("://")[-1].split("/")[0].split(":")[0]
    
    if origin_host in ["localhost", "127.0.0.1", "::1"]:
        return True
        
    try:
        ip = ipaddress.ip_address(origin_host)
        return ip.is_private
    except ValueError:
        # 解析失败说明是域名（如内网配置的 domain.local），开发环境直接放行
        return True

@web.middleware
async def localCors_middleware(request, handler):
    # 1. 拦截预检请求 (OPTIONS)
    if request.method == 'OPTIONS':
        origin = request.headers.get('Origin')
        if is_local_origin(origin):
            return web.Response(
                status=204,
                headers={
                    'Access-Control-Allow-Origin': origin,
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': '*',  # 允许所有请求头
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Max-Age': '86400',   # 预检缓存1天
                }
            )
        return web.Response(status=403)  # 非内网来源直接拒绝预检

    # 2. 处理正常业务请求
    response = await handler(request)
    
    origin = request.headers.get('Origin')
    if is_local_origin(origin):
        # 注意：这里必须返回具体的 Origin 字符串，绝对不能返回 "*"，否则浏览器会拒绝
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        
    return response
def main():
    global config, llm_service, virtualcam_quit_event, virtualcam_render_thread
    
    # 1. 加载配置
    config = AppConfig.from_args()
    logger.info(f"Configuration loaded: {config}")

    # 2. 初始化进程模式
    mp.set_start_method('spawn')
    
    # 3. 初始化 LLM 服务
    try:
        llm_service = create_llm_service(config)
    except Exception as e:
        logger.warning(f"LLM Service initialization failed: {e}")

    # 4. 预加载模型
    if config.server.transport == 'virtualcam':
        init_model()
        
        logger.info("Running in VirtualCam mode...")
        
        config.session_id = 0
        
        if config.model.name == 'musetalk':
             render_cls = MuseRenderLoop
        elif config.model.name == 'wav2lip':
             render_cls = LipRenderLoop
        elif config.model.name == 'ultralight':
             render_cls = LightRenderLoop
        else: raise ValueError("Unknown model")
        
        render_loop = render_cls(config, model_instance, avatar_instance)
        
        from threading import Event
        thread_quit = Event()
        virtualcam_quit_event = thread_quit  # 保存到全局变量
        
        import threading
        render_thread = threading.Thread(target=render_loop.render, args=(thread_quit,))
        render_thread.daemon = False  # 设置为非守护线程，确保能正确等待
        render_thread.start()
        virtualcam_render_thread = render_thread  # 保存到全局变量
        
        logger.info("VirtualCam render thread started.")
        
        # 等待退出信号
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, shutting down VirtualCam...")
            thread_quit.set()
            render_thread.join(timeout=5)
            logger.info("VirtualCam stopped.")
        
        return  # 虚拟摄像头模式直接退出
        
    else:
        # WebRTC / RTCPush 模式：预加载模型
        init_model()
    
    # 5. 构建 Web 应用 (注入自定义中间件)
    middlewares = [localCors_middleware] # 刚才定义的中间件名
    app = web.Application(client_max_size=1024**2 * 100, middlewares=middlewares)
    app.on_shutdown.append(on_shutdown)

    # 6. 注册路由
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    
    app.router.add_static('/', path='web')

    # 7. 启动服务
    pagename = 'webrtcapi.html'
    if config.server.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif config.server.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
        
    logger.info(f"Server running at http://{config.server.host}:{config.server.listen_port}/{pagename}")
    logger.info(f"If using WebRTC, visit: http://{config.server.host}:{config.server.listen_port}/dashboard.html")

    run_server(web.AppRunner(app))

if __name__ == '__main__':
    main()