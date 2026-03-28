from typing import Any, Optional
from configs import ModelConfig
from logger import logger

from .base_asr import BaseASR
from .muse_asr import MuseASR
from .hubert_asr import HubertASR
from .lip_asr import LipASR

def create_asr_service(config: ModelConfig, parent_ref: Any, audio_processor: Any = None) -> BaseASR:
    """
    创建 ASR 服务的工厂函数
    :param config: 模型配置
    :param parent_ref: 父类实例
    :param audio_processor: 音频处理器实例 (Audio2Feature)
                            MuseTalk/UL 需要此参数，Wav2Lip 不需要 (内部使用 wav2lip.audio)
    """
    model_name = config.name.lower()
    
    if model_name == 'musetalk':
        logger.info("Initializing ASR: MuseASR (Whisper)")
        if audio_processor is None: raise ValueError("MuseASR requires audio_processor")
        return MuseASR(config, parent_ref, audio_processor)
    
    elif model_name == 'wav2lip':
        logger.info("Initializing ASR: LipASR (Mel-Spectrogram)")
        # LipASR 不需要外部的 audio_processor，传 None 即可
        return LipASR(config, parent_ref)
    
    elif model_name == 'ultralight':
        logger.info("Initializing ASR: HubertASR")
        if audio_processor is None: raise ValueError("HubertASR requires audio_processor")
        return HubertASR(config, parent_ref, audio_processor)
    
    else:
        raise ValueError(f"Unsupported model for ASR: {model_name}")

__all__ = ['BaseASR', 'create_asr_service']