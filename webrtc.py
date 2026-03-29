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

import asyncio
import fractions
import time
import threading
from typing import Tuple, Optional, Set, Union

import numpy as np
from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet
from aiortc import MediaStreamTrack

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
from logger import logger as mylogger

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040  # 25fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

class PlayerStreamTrack(MediaStreamTrack):
    """
    新架构下的轨道类：
    - 仅作为消费者，从队列中获取已处理好的帧。
    - 不再反向调用 container.notify。
    - 时间戳管理保持不变。
    """

    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue(maxsize=100)
        
        self.timelist = []
        self.current_frame_count = 0
        
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                mylogger.info('video start:%f', self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else:  # audio
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                mylogger.info('audio start:%f', self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        """
        接收数据：仅从队列获取 (frame, eventpoint)
        """
        self._player._start(self)
        
        # 获取数据
        frame, eventpoint = await self._queue.get()
        
        # 设置时间戳
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        
        # 注意：此处移除了 notify 调用。
        # eventpoint 在新架构中已由 RenderLoop.process_frames 处理（用于控制 recorder 等），
        # 轨道层不再需要关心业务事件。
        
        if frame is None:
            self.stop()
            raise Exception("Frame is None")

        # 统计视频帧率
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
                
        return frame

    def stop(self):
        super().stop()
        # 清空队列
        while not self._queue.empty():
            self._queue.get_nowait()
        
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    loop,
    container,
    audio_track,
    video_track
):
    """
    工作线程入口：直接调用 RenderLoop.render
    """
    container.render(quit_event, loop, audio_track, video_track)

class HumanPlayer:
    """
    适配器：连接 WebRTC 轨道与 RenderLoop
    """
    def __init__(self, nerfreal, loop: asyncio.AbstractEventLoop):
        self.__loop = loop
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__started: Set[PlayerStreamTrack] = set()
        
        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        
        # nerfreal 现在是 RenderLoop 实例
        self.__container = nerfreal

    @property
    def audio(self) -> MediaStreamTrack:
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        
        # 只在首次启动时创建渲染线程
        if self.__thread is None:
            mylogger.debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__loop,
                    self.__container,
                    self.__audio,
                    self.__video
                ),
                daemon=True # 设置为守护线程，主程序退出时自动结束
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            mylogger.debug("Stopping worker thread")
            self.__thread_quit.set()
            # self.__thread.join() # 守护线程通常不需要 join，或者视具体停止逻辑而定
            self.__thread = None

        # 注意：不要在这里销毁 __container (RenderLoop)，因为它可能被其他地方引用
        # 生命周期应由 SessionManager 管理

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)