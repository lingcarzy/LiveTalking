import time
import numpy as np
from typing import List, Any

from .base_asr import BaseASR
from configs import ModelConfig
from logger import logger

# 引入 wav2lip 的音频处理工具
# 假设项目中存在 wav2lip/audio.py
try:
    from wav2lip import audio
except ImportError:
    logger.error("wav2lip.audio module not found. Please ensure wav2lip directory is in the path.")
    audio = None

class LipASR(BaseASR):
    def __init__(self, config: ModelConfig, parent_ref: Any):
        super().__init__(config, parent_ref)
        if audio is None:
            raise RuntimeError("LipASR requires wav2lip.audio module.")
        
        # Wav2Lip 特有的参数
        # mel_step_size 通常对应模型的输入尺寸，Wav2Lip 默认为 16 (对应约 0.2秒音频? 取决于 hop size)
        self.mel_step_size = 16 

    def run_step(self):
        start_time = time.time()
        
        # 1. 收集音频帧 (batch_size * 2 个 chunk)
        for _ in range(self.batch_size * 2):
            frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(frame)
            self.output_queue.put((frame, type, eventpoint))
        
        # 上下文不足则返回
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # 2. 拼接帧
        inputs = np.concatenate(self.frames) # [N * chunk]
        
        # 3. 计算 Mel 频谱
        # audio.melspectrogram 内部通常包含 STFT 计算
        mel = audio.melspectrogram(inputs)
        
        # 4. 分块处理 (复杂的对齐逻辑，保留原逻辑核心)
        # 原代码逻辑：
        # left = max(0, self.stride_left_size * 80 / 50)
        # 这里的 80/50 可能与帧率和 Mel 的 hop size 有关
        # Wav2Lip 默认采样率 16000, hop_size 160 (10ms), 所以 1秒 100帧
        # 视频帧率 fps=50, 所以 1个视频帧 = 2个 Mel 帧
        # stride_left_size 是视频帧单位，对应的 Mel 帧数 = stride * 2
        
        left = max(0, self.stride_left_size * 80 / 50)
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size * 80 / 50)
        
        mel_idx_multiplier = 80. * 2 / self.fps 
        mel_step_size = 16
        i = 0
        mel_chunks = []
        
        # 循环生成 batch 个 mel chunks
        # 原代码条件：while i < (len(self.frames)-self.stride_left_size-self.stride_right_size)/2:
        # 这里的 /2 是因为 batch_size * 2 的关系吗？保持原逻辑结构
        processed_frames_count = len(self.frames) - self.stride_left_size - self.stride_right_size
        while i < processed_frames_count / 2:
            start_idx = int(left + i * mel_idx_multiplier)
            
            if start_idx + mel_step_size > len(mel[0]):
                # 边界处理：如果超出范围，取最后一段
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
            
        # 5. 送入特征队列
        self.feat_queue.put(mel_chunks)
        
        # 6. 维护滑动窗口
        #self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        
        # logger.debug(f"LipASR step cost: {(time.time() - start_time) * 1000}ms")