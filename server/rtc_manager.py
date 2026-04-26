###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
import time
from typing import Dict, Optional
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()
        self._pc_sessions: Dict[RTCPeerConnection, str] = {}

    def _release_session_for_pc(self, pc: RTCPeerConnection) -> None:
        sessionid = self._pc_sessions.pop(pc, None)
        if sessionid:
            session_manager.remove_session(sessionid)

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        offer_start = time.perf_counter()
        pc: Optional[RTCPeerConnection] = None
        sessionid: Optional[str] = None
        params = await request.json()

        offer_sdp = params.get("sdp")
        offer_type = params.get("type")
        if not isinstance(offer_sdp, str) or not offer_sdp.strip() or not isinstance(offer_type, str):
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "invalid offer payload"}),
            )

        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)

        if False: # 不再由 RTCManager 控制 max_session，让业务逻辑或SessionManager 控制
            logger.info('reach max session')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        #sessionid = _rand_session_id()

        try:
            # 通过 SessionManager 构建
            sessionid = await session_manager.create_session(params)
            logger.info('offer sessionid=%s', sessionid)
            avatar_session = session_manager.get_session(sessionid)
            if avatar_session is None:
                raise RuntimeError(f'session build failed: {sessionid}')

            # 创建 PeerConnection
            ice_urls = [u.strip() for u in str(getattr(self.opt, 'ice_server_urls', '')).split(',') if u.strip()]
            if ice_urls:
                ice_servers = [RTCIceServer(urls=u) for u in ice_urls]
                pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
                logger.info('offer using ICE servers: %s', ice_urls)
            else:
                pc = RTCPeerConnection()
                logger.info('offer using no ICE server (LAN/local mode)')
            self.pcs.add(pc)
            self._pc_sessions[pc] = sessionid

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info("Connection state is %s", pc.connectionState)
                if pc.connectionState in ("failed", "closed", "disconnected"):
                    await pc.close()
                    self.pcs.discard(pc)
                    self._release_session_for_pc(pc)

            # 添加发送轨道
            from server.webrtc import HumanPlayer
            player = HumanPlayer(avatar_session)
            pc.addTrack(player.audio)
            pc.addTrack(player.video)

            # 设置编解码器偏好
            capabilities = RTCRtpSender.getCapabilities("video")
            preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
            video_transceiver = next((t for t in pc.getTransceivers() if getattr(t, 'kind', '') == 'video'), None)
            if video_transceiver is not None and preferences:
                video_transceiver.setCodecPreferences(preferences)

            await pc.setRemoteDescription(offer)

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            logger.info('offer done sessionid=%s elapsed=%.3fs', sessionid, time.perf_counter() - offer_start)

            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "sessionid": sessionid,
                }),
            )
        except Exception as e:
            logger.exception('handle_offer failed')
            if pc is not None:
                try:
                    await pc.close()
                except Exception:
                    logger.exception('pc close failed during offer rollback')
                self.pcs.discard(pc)
                self._release_session_for_pc(pc)
            elif sessionid:
                session_manager.remove_session(sessionid)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"offer failed: {e}"}),
            )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        pc: Optional[RTCPeerConnection] = None
        try:
            await session_manager.create_session({}, sessionid)
            avatar_session = session_manager.get_session(sessionid)
            if avatar_session is None:
                raise RuntimeError(f'session build failed: {sessionid}')

            pc = RTCPeerConnection()
            self.pcs.add(pc)
            self._pc_sessions[pc] = sessionid

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info("Connection state is %s", pc.connectionState)
                if pc.connectionState in ("failed", "closed", "disconnected"):
                    await pc.close()
                    self.pcs.discard(pc)
                    self._release_session_for_pc(pc)

            from server.webrtc import HumanPlayer
            player = HumanPlayer(avatar_session)
            pc.addTrack(player.audio)
            pc.addTrack(player.video)

            await pc.setLocalDescription(await pc.createOffer())

            async with aiohttp.ClientSession() as session:
                async with session.post(push_url, data=pc.localDescription.sdp) as response:
                    answer_sdp = await response.text()

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type='answer')
            )
        except Exception:
            logger.exception('handle_rtcpush failed sessionid=%s', sessionid)
            if pc is not None:
                try:
                    await pc.close()
                except Exception:
                    logger.exception('pc close failed during rtcpush rollback')
                self.pcs.discard(pc)
                self._release_session_for_pc(pc)
            else:
                session_manager.remove_session(sessionid)
            raise

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        for pc in list(self._pc_sessions.keys()):
            self._release_session_for_pc(pc)
        self.pcs.clear()
