import time
import asyncio
import numpy as np
import soundfile as sf
from io import BytesIO
import edge_tts

from .base_tts import BaseTTS, State
from configs import TTSConfig
from logger import logger

class EdgeTTS(BaseTTS):
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        self.voicename = config.ref_file # e.g., "zh-CN-YunxiaNeural"

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t = time.time()
        
        # 运行异步任务
        asyncio.new_event_loop().run_until_complete(self._generate_audio(text))
        logger.info(f'EdgeTTS generation time: {time.time()-t:.4f}s')

        if self.input_stream.getbuffer().nbytes <= 0:
            logger.error('EdgeTTS generated empty audio!')
            return

        # 处理音频流
        self.input_stream.seek(0)
        stream, sample_rate = sf.read(self.input_stream)
        stream = stream.astype(np.float32)
        
        if stream.ndim > 1:
            stream = stream[:, 0]
            
        stream = self.resample_audio(stream, sample_rate, self.sample_rate)
        self._push_audio_stream(stream, msg)
        
        # 清空缓存
        self.input_stream.seek(0)
        self.input_stream.truncate()

    async def _generate_audio(self, text: str):
        try:
            communicate = edge_tts.Communicate(text, self.voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    self.input_stream.write(chunk["data"])
        except Exception as e:
            logger.exception('EdgeTTS generation error')