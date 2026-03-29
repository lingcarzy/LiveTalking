#service/tts/base_tts.py
import time
import numpy as np
import resampy
import queue
from queue import Queue
from enum import Enum
from threading import Thread, Event
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Iterator

# 引入配置类
from configs import TTSConfig
from logger import logger

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS(ABC):
    def __init__(self, config: TTSConfig, parent_ref: Any):
        """
        :param config: TTS 配置对象
        :param parent_ref: 父类实例引用，用于调用 put_audio_frame
        """
        self.config = config
        self.parent = parent_ref

        # 音频参数
        self.fps = 50  # 默认 20ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        """清空队列并暂停当前播放"""
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def render(self, quit_event: Event):
        """启动 TTS 处理线程"""
        actual_event = quit_event if quit_event is not None else self.parent._quit_event
        process_thread = Thread(target=self.process_tts, args=(actual_event,))
        process_thread.start()

    def process_tts(self, quit_event: Event):
        while not quit_event.is_set():
            try:
                msg: Tuple[str, Dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            
            try:
                self.txt_to_audio(msg)
            except Exception as e:
                logger.exception(f"TTS processing error: {e}")
        
        logger.info('TTS process thread stopped')

    @abstractmethod
    def txt_to_audio(self, msg: Tuple[str, Dict]):
        """
        核心抽象方法：子类需实现此方法将文本转换为音频帧
        并调用 self.parent.put_audio_frame 推送数据
        """
        pass

    # ------------------- 辅助方法 -------------------
    
    def _push_audio_stream(self, audio_stream: np.ndarray, msg: Tuple[str, Dict], 
                           is_first: bool = True, is_last: bool = True):
        """
        通用的音频流分块推送逻辑
        :param audio_stream: 已经重采样到 16kHz 的 float32 numpy 数组
        """
        text, textevent = msg
        streamlen = audio_stream.shape[0]
        idx = 0
        
        while streamlen >= self.chunk:
            if self.state != State.RUNNING:
                return

            eventpoint = {}
            if is_first and idx == 0:
                eventpoint = {'status': 'start', 'text': text}
                eventpoint.update(**textevent)
            
            self.parent.put_audio_frame(audio_stream[idx:idx + self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

        # 发送结束标记
        if is_last:
            eventpoint = {'status': 'end', 'text': text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    @staticmethod
    def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr != target_sr and audio_data.shape[0] > 0:
            return resampy.resample(x=audio_data, sr_orig=orig_sr, sr_new=target_sr)
        return audio_data