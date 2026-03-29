# core/session_manager.py
import asyncio
import gc
import random
from typing import Dict, Optional, Any

from configs import AppConfig
from core.render_loop import RenderLoop
from logger import logger

class SessionManager:
    def __init__(self):
        self._sessions: Dict[int, RenderLoop] = {}
        self._lock = asyncio.Lock()

    def _generate_id(self) -> int:
        """生成 6 位随机 ID"""
        return random.randint(100000, 999999)
    async def register_session(self, session_id: int, render_loop: RenderLoop):
        """外部安全注册会话（主要用于RTCPush）"""
        async with self._lock:
            self._sessions[session_id] = render_loop
    async def create_session(self, config: AppConfig, model_instance: Any, avatar_instance: Any) -> RenderLoop:
        """
        创建新的渲染会话
        注意：这里不再负责加载模型，而是接收已加载的模型实例
        """
        async with self._lock:
            # 生成唯一 ID
            session_id = self._generate_id()
            while session_id in self._sessions:
                session_id = self._generate_id()
            
            config.session_id = session_id
            
            # 工厂模式：根据配置选择具体的 RenderLoop 类
            if config.model.name == 'musetalk':
                from core.muse_render import MuseRenderLoop
                render_class = MuseRenderLoop
            elif config.model.name == 'wav2lip':
                from core.lip_render import LipRenderLoop
                render_class = LipRenderLoop
            elif config.model.name == 'ultralight':
                from core.light_render import LightRenderLoop
                render_class = LightRenderLoop
            else:
                raise ValueError(f"Unsupported model: {config.model.name}")

            logger.info(f"Creating session {session_id} with {render_class.__name__}")
            
            # 实例化 RenderLoop (这是一个同步操作，但在 async with lock 下执行是安全的)
            render_loop = render_class(
                config=config,
                model_instance=model_instance,
                avatar_data=avatar_instance
            )
            
            self._sessions[session_id] = render_loop
            return render_loop

    async def destroy_session(self, session_id: int):
        """销毁会话并清理资源"""
        async with self._lock:
            if session_id in self._sessions:
                logger.info(f"Destroying session {session_id}")
                session = self._sessions.pop(session_id)
                
                # 如果 RenderLoop 有 cleanup 方法，在此调用
                # 当前架构中，资源清理主要由 python gc 和 session 结束自动处理
                # 如果有显式关闭 FFmpeg 管道等需求，可实现 session.cleanup()
                if hasattr(session, 'cleanup'):
                    await session.cleanup()
                try:
                    if hasattr(session, 'stop'):
                        session.stop()
                except Exception as e:
                    logger.error(f"Error stopping session {session_id}: {e}")
                del session
                gc.collect()

    def get_session(self, session_id) -> Optional[RenderLoop]:
        # 强制转换类型，防止 int/str 不匹配
        try:
            sid = int(session_id)
        except (ValueError, TypeError):
            return None
            
        return self._sessions.get(sid)

# 全局单例
session_manager = SessionManager()