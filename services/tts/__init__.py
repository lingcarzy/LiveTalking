from typing import Any
from configs import TTSConfig
from logger import logger

# Import implementations
from .base_tts import BaseTTS
from .edge_tts import EdgeTTS
from .fish_tts import FishTTS
from .sovits_tts import SovitsTTS
from .cosyvoice_tts import CosyVoiceTTS
from .tencent_tts import TencentTTS
from .doubao_tts import DoubaoTTS
from .index_tts import IndexTTS2
from .xtts import XTTS
from .azure_tts import AzureTTS

def create_tts_service(config: TTSConfig, parent_ref: Any) -> BaseTTS:
    """Factory function to create TTS instance"""
    engine = config.engine.lower()
    
    tts_map = {
        "edgetts": EdgeTTS,
        "fishtts": FishTTS,
        "gpt-sovits": SovitsTTS,
        "cosyvoice": CosyVoiceTTS,
        "tencent": TencentTTS,
        "doubao": DoubaoTTS,
        "indextts2": IndexTTS2,
        "xtts": XTTS,
        "azuretts": AzureTTS
    }
    
    if engine not in tts_map:
        raise ValueError(f"Unsupported TTS engine: {engine}")
    
    logger.info(f"Initializing TTS engine: {engine}")
    return tts_map[engine](config, parent_ref)

__all__ = ['BaseTTS', 'create_tts_service']