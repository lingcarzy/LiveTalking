# app.py
import sys
import os

# 将当前文件所在的目录（项目根目录）添加到系统路径中
# 这样 Python 就能找到 configs, core, services 等文件夹了
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import gc
import json
import random
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
# 全局配置对象
config: Optional[AppConfig] = None
llm_service = None
# 全局模型实例 (暂时保留，避免重复加载)
model_instance = None
avatar_instance = None

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
    # 注意：这里的 session 创建逻辑与 WebRTC offer 不同
    # 我们直接调用 SessionManager 的内部逻辑或工厂方法
    config.session_id = sessionid
    
    # 简单起见，直接实例化 (或者调用 session_manager.create_session 的同步版本)
    # 这里为了保持一致性，我们手动构建
    if config.model.name == 'musetalk':
         render_cls = MuseRenderLoop
    elif config.model.name == 'wav2lip':
         render_cls = LipRenderLoop
    elif config.model.name == 'ultralight':
         render_cls = LightRenderLoop
    else: return

    render_loop = render_cls(config, model_instance, avatar_instance)
    # 将其加入管理 (可选，如果需要后续控制)
    session_manager._sessions[sessionid] = render_loop

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Push Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            await session_manager.destroy_session(sessionid)

    player = HumanPlayer(render_loop)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    
    if answer:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))
    else:
        logger.error(f"Failed to get answer from {push_url}")
# =============================================================================
# 路由处理函数
# =============================================================================

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    # 1. 使用 SessionManager 创建会话
    try:
        nerfreal = await session_manager.create_session(config, model_instance, avatar_instance)
        sessionid = nerfreal.session_id
    except Exception as e:
        logger.exception("Failed to create session")
        return web.json_response({"code": -1, "msg": str(e)})

    logger.info(f'New session created: {sessionid}, total sessions: {len(session_manager._sessions)}')

    # WebRTC 配置
    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Session {sessionid} - Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            # 2. 使用 SessionManager 销毁会话
            await session_manager.destroy_session(sessionid)

    player = HumanPlayer(nerfreal)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # 编解码器偏好设置
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
        
        # 使用 SessionManager 获取会话
        session = session_manager.get_session(sessionid)
        if not session:
            return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        if params.get('interrupt'):
            session.flush_talk()
        
        if params['type'] == 'echo':
            session.put_msg_txt(params['text'])
        elif params['type'] == 'chat':
            # 定义异步处理流程
            async def process_chat():
                try:
                    # 使用 async for 迭代异步生成器
                    async for text_segment in llm_service.chat_stream(params['text']):
                        session.put_msg_txt(text_segment)
                except Exception as e:
                    logger.error(f"LLM chat stream error: {e}")
            
            # 定义一个同步的包装函数，用于在 executor 中运行
            def run_async_chat():
                # 获取当前线程的事件循环，如果没有则创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(process_chat())
                finally:
                    loop.close()

            # 在线程池中运行同步包装函数
            if llm_service:
                asyncio.get_event_loop().run_in_executor(None, run_async_chat)
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
    """处理上传的音频文件"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        # filename = fileobj.filename # 暂时未使用
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
    """设置自定义音频/视频状态"""
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
    """控制录制开始/结束"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        
        session = session_manager.get_session(sessionid)
        if not session:
             return web.json_response({"code": -1, "msg": "Session not found"}, status=404)

        if params['type'] == 'start_record':
            # 获取当前视频帧分辨率用于初始化 recorder
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
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

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

def main():
    global config, llm_service
    
    # 1. 加载配置
    config = AppConfig.from_args()
    logger.info(f"Configuration loaded: {config}")

    # 2. 初始化进程模式
    mp.set_start_method('spawn')
    
    # 3. 初始化 LLM 服务 (即使 API Key 为空也初始化，防止报错，内部会处理)
    try:
        llm_service = create_llm_service(config)
    except Exception as e:
        logger.warning(f"LLM Service initialization failed: {e}")

    # 4. 预加载模型
    if config.server.transport == 'virtualcam':
        # VirtualCam 模式：需要先加载模型，并在主线程外启动渲染
        init_model()
        
        # 创建一个会话实例
        # 注意：virtualcam 模式通常只需要一个实例，session_id 设为 0
        logger.info("Running in VirtualCam mode...")
        
        # 使用 SessionManager 创建会话
        # 这里需要在异步环境外调用，或者手动构建
        # 为简化，我们直接构建 RenderLoop
        # 注意：VirtualCam 模式通常不经过 WebRTC offer 流程
        
        # 由于 SessionManager.create_session 是 async 的，这里做个适配
        # 或者直接调用工厂逻辑
        config.session_id = 0
        
        # 选择渲染类
        if config.model.name == 'musetalk':
             render_cls = MuseRenderLoop
        elif config.model.name == 'wav2lip':
             render_cls = LipRenderLoop
        elif config.model.name == 'ultralight':
             render_cls = LightRenderLoop
        else: raise ValueError("Unknown model")
        
        render_loop = render_cls(config, model_instance, avatar_instance)
        
        # 启动渲染线程
        from threading import Event
        thread_quit = Event()
        # 启动 render 方法 (注意 render 方法是阻塞的，或者内部启动线程)
        # 原 BaseReal.render 是启动线程，但我们的 RenderLoop.render 是阻塞循环
        # 我们需要将其放在独立线程中运行
        import threading
        render_thread = threading.Thread(target=render_loop.render, args=(thread_quit,))
        render_thread.daemon = True
        render_thread.start()
        
        logger.info("VirtualCam render thread started.")
        pass

    else:
        # WebRTC / RTCPush 模式：预加载模型
        init_model()
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', config.server.listen_port)
        loop.run_until_complete(site.start())
        
        # 如果是 rtcpush 模式，启动推流
        if config.server.transport == 'rtcpush':
            for k in range(config.server.max_session):
                push_url = config.server.push_url
                if k != 0:
                    push_url = config.server.push_url + str(k)
                # 在事件循环中启动推流任务
                loop.run_until_complete(run(push_url, k))
                
        loop.run_forever()
    # 5. 构建 Web 应用
    app = web.Application(client_max_size=1024**2 * 100) # 100MB upload limit
    app.on_shutdown.append(on_shutdown)

    # 6. 注册路由
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    
    # 静态文件目录
    app.router.add_static('/', path='web')

    # 7. CORS 配置
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    # 8. 启动服务
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