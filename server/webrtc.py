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
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = queue.Queue(maxsize=100)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
        self.recv_wait_total = 0.0
        self.recv_wait_count = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        qwait_start = time.perf_counter()
        frame, eventpoint = await asyncio.to_thread(self._queue.get)
        qwait = time.perf_counter() - qwait_start

        self.recv_wait_total += qwait
        self.recv_wait_count += 1
        if self._player is not None:
            self._player._on_track_recv(self.kind, qwait)
                
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        # Drain & delete remaining frames
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    container
):
    container.render(quit_event)

class HumanPlayer:

    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self

        self._stats = {
            'video_pushed': 0,
            'audio_pushed': 0,
            'video_dropped': 0,
            'audio_dropped': 0,
            'video_recv': 0,
            'audio_recv': 0,
            'video_wait_total': 0.0,
            'audio_wait_total': 0.0,
        }
        self._stats_last_time = time.perf_counter()
        self._stats_last_video_pushed = 0
        self._stats_last_video_recv = 0
        self._stats_last_audio_pushed = 0
        self._stats_last_audio_recv = 0
        self._last_video_frame = None
        self._video_shape = None

    @staticmethod
    def _push_with_drop(q: queue.Queue, item) -> int:
        """Keep real-time behavior by dropping oldest frame when the queue is full."""
        dropped = 0
        while True:
            try:
                q.put_nowait(item)
                return dropped
            except queue.Full:
                try:
                    q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    return dropped

    def push_video(self, frame):
        from av import VideoFrame
        self._video_shape = frame.shape
        self._last_video_frame = frame.copy()
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        dropped = self._push_with_drop(self.__video._queue, (new_frame, None))
        self._stats['video_pushed'] += 1
        self._stats['video_dropped'] += dropped
        self._maybe_log_stats()

    def push_audio(self, frame, eventpoint=None):
        from av import AudioFrame
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = 16000
        dropped = self._push_with_drop(self.__audio._queue, (new_frame, eventpoint))
        self._stats['audio_pushed'] += 1
        self._stats['audio_dropped'] += dropped
        self._maybe_log_stats()

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

    def notify(self,eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    def _on_track_recv(self, kind: str, wait_seconds: float) -> None:
        if kind == 'video':
            self._stats['video_recv'] += 1
            self._stats['video_wait_total'] += wait_seconds
        else:
            self._stats['audio_recv'] += 1
            self._stats['audio_wait_total'] += wait_seconds
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._stats_last_time
        if elapsed < 5.0:
            return

        dv_push = self._stats['video_pushed'] - self._stats_last_video_pushed
        dv_recv = self._stats['video_recv'] - self._stats_last_video_recv
        da_push = self._stats['audio_pushed'] - self._stats_last_audio_pushed
        da_recv = self._stats['audio_recv'] - self._stats_last_audio_recv
        v_wait_avg_ms = 0.0
        a_wait_avg_ms = 0.0
        if self._stats['video_recv'] > 0:
            v_wait_avg_ms = self._stats['video_wait_total'] * 1000.0 / self._stats['video_recv']
        if self._stats['audio_recv'] > 0:
            a_wait_avg_ms = self._stats['audio_wait_total'] * 1000.0 / self._stats['audio_recv']

        mylogger.info(
            'webrtc queue stats: vq=%d aq=%d v_push_fps=%.2f v_recv_fps=%.2f a_push=%.2f a_recv=%.2f v_drop=%d a_drop=%d v_wait=%.2fms a_wait=%.2fms',
            self.__video._queue.qsize(),
            self.__audio._queue.qsize(),
            dv_push / elapsed,
            dv_recv / elapsed,
            da_push / elapsed,
            da_recv / elapsed,
            self._stats['video_dropped'],
            self._stats['audio_dropped'],
            v_wait_avg_ms,
            a_wait_avg_ms,
        )

        self._stats_last_time = now
        self._stats_last_video_pushed = self._stats['video_pushed']
        self._stats_last_video_recv = self._stats['video_recv']
        self._stats_last_audio_pushed = self._stats['audio_pushed']
        self._stats_last_audio_recv = self._stats['audio_recv']

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None and len(self.__started) >= 2:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)

    def get_debug_stats(self) -> dict:
        return {
            'video_queue': self.__video._queue.qsize(),
            'audio_queue': self.__audio._queue.qsize(),
            **self._stats,
        }
