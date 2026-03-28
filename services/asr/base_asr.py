import time
import numpy as np
import queue
from queue import Queue
import torch.multiprocessing as mp
from typing import Optional, Tuple, Any, List
from abc import ABC, abstractmethod
from collections import deque

from configs import ModelConfig
from logger import logger

class BaseASR(ABC):
    def __init__(self, config: ModelConfig, parent_ref: Any = None):
        """
        :param config: 模型配置对象
        :param parent_ref: 父类实例引用，用于获取自定义音频流
        """
        self.config = config
        self.parent = parent_ref

        # 音频参数
        self.fps = config.fps  # 通常是 50 (20ms per frame)
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples
        
        # 输入输出队列
        self.queue = Queue()           # 接收音频帧 (来自 TTS 或 麦克风)
        self.output_queue = mp.Queue() # 输出音频帧 (用于播放或录音)
        self.feat_queue = mp.Queue(2)  # 输出特征向量 (用于模型推理)

        # 缓冲区参数
        self.batch_size = config.batch_size
        self.stride_left_size = getattr(config, 'l', 10) # 上下文填充
        self.stride_right_size = getattr(config, 'r', 10)
        max_len = (self.stride_left_size + self.stride_right_size) + (self.batch_size * 2)
        self.frames = deque(maxlen=max_len) 

    def flush_talk(self):
        """清空输入队列"""
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk: np.ndarray, datainfo: dict = {}):
        """外部接口：放入音频帧"""
        self.queue.put((audio_chunk, datainfo))

    def get_audio_frame(self) -> Tuple[np.ndarray, int, Optional[dict]]:
        """
        获取单帧音频
        :return: (audio_frame, type, eventpoint)
                 type: 0-正常语音, 1-静音, >1-自定义音频
        """
        try:
            frame, eventpoint = self.queue.get(block=True, timeout=0.01)
            return frame, 0, eventpoint
        except queue.Empty:
            # 如果没有输入，检查父类是否有自定义音频流（如背景音乐、特定动作音频）
            if self.parent and hasattr(self.parent, 'curr_state') and self.parent.curr_state > 1:
                if hasattr(self.parent, 'get_audio_stream'):
                    frame = self.parent.get_audio_stream(self.parent.curr_state)
                    return frame, self.parent.curr_state, None
            
            # 否则返回静音帧
            return np.zeros(self.chunk, dtype=np.float32), 1, None

    def get_audio_out(self) -> Tuple[np.ndarray, int, Optional[dict]]:
        """获取处理后的音频帧（用于播放）"""
        return self.output_queue.get()

    def warm_up(self):
        """预热缓冲区，填充初始帧"""
        logger.info("Warming up ASR buffer...")
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type, eventpoint))
        
        # 清除部分输出队列，保持延迟一致性
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    @abstractmethod
    def run_step(self):
        """
        核心处理逻辑，由子类实现
        1. 收集 batch_size 个帧
        2. 提取特征
        3. 放入 feat_queue
        """
        pass

    def get_next_feat(self, block=True, timeout=None):
        """推理端获取特征"""
        return self.feat_queue.get(block, timeout)