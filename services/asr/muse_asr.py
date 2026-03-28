import time
import numpy as np
from .base_asr import BaseASR
from configs import ModelConfig
from logger import logger

# 延迟导入，避免循环依赖或加载过重
Audio2Feature = None

class MuseASR(BaseASR):
    def __init__(self, config: ModelConfig, parent_ref, audio_processor):
        """
        :param audio_processor: 预初始化的 Audio2Feature 实例
        """
        super().__init__(config, parent_ref)
        self.audio_processor = audio_processor

    def run_step(self):
        start_time = time.time()
        
        # 1. 收集音频帧 (batch_size * 2 个 chunk)
        # 乘以2是因为通常音频采样策略或特征提取步长需求
        for _ in range(self.batch_size * 2):
            audio_frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            # 原始音频帧直接输出，用于播放
            self.output_queue.put((audio_frame, type, eventpoint))
        
        # 确保缓冲区有足够的数据进行特征提取
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # 2. 拼接帧并进行特征提取
        inputs = np.concatenate(self.frames)  # [N * chunk]
        
        # Whisper 特征提取
        # audio2feat 内部会处理 resample 等逻辑
        whisper_feature = self.audio_processor.audio2feat(inputs)
        
        # 3. 分块处理，适配视频帧率
        # fps/2 是因为音频特征通常对应半个视频帧或者其他对齐策略
        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_feature, 
            fps=self.fps / 2, 
            batch_size=self.batch_size, 
            start=self.stride_left_size / 2
        )
        
        # 4. 送入特征队列供推理使用
        self.feat_queue.put(whisper_chunks)
        
        # 5. 丢弃旧数据，保持滑动窗口
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        
        # logger.debug(f"MuseASR step cost: {(time.time() - start_time) * 1000}ms")