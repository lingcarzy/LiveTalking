import time
import numpy as np
from .base_asr import BaseASR
from configs import ModelConfig
from logger import logger

class HubertASR(BaseASR):
    def __init__(self, config: ModelConfig, parent_ref, audio_processor, audio_feat_length=[8, 8]):
        """
        :param audio_processor: 预初始化的 Ultralight Audio2Feature 实例
        :param audio_feat_length: 特征上下文长度 [left, right]
        """
        super().__init__(config, parent_ref)
        self.audio_processor = audio_processor
        self.audio_feat_length = audio_feat_length

    def run_step(self):
        start_time = time.time()
        
        # 1. 收集音频帧
        for _ in range(self.batch_size * 2):
            audio_frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type, eventpoint))
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames)  # [N * chunk]

        # 2. HuBERT 特征提取
        mel = self.audio_processor.get_hubert_from_16k_speech(inputs)
        
        # 3. 分块
        mel_chunks = self.audio_processor.feature2chunks(
            feature_array=mel, 
            fps=self.fps / 2, 
            batch_size=self.batch_size, 
            audio_feat_length=self.audio_feat_length, 
            start=self.stride_left_size / 2
        )

        # 4. 送入队列
        self.feat_queue.put(mel_chunks)
        
        # 5. 维护滑动窗口
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        # logger.debug(f"HubertASR step cost: {(time.time() - start_time) * 1000}ms")