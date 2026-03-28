import time
import numpy as np
import soundfile as sf
import requests
from io import BytesIO

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class SovitsTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]): 
        text, textevent = msg
        audio_iter = self._fetch_stream(text)
        self._process_stream(audio_iter, msg)

    def _fetch_stream(self, text: str):
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': "zh",
            'ref_audio_path': self.config.ref_file,
            'prompt_text': self.config.ref_text,
            'prompt_lang': "zh",
            'media_type': 'ogg',
            'streaming_mode': True
        }
        
        try:
            res = requests.post(f"{self.config.server}/tts", json=req, stream=True)
            if res.status_code != 200:
                logger.error(f"SovitsTTS Error: {res.text}")
                return

            for chunk in res.iter_content(chunk_size=None):
                if chunk and self.state == BaseTTS.State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('SovitsTTS request error')

    def _process_stream(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk:
                # Sovits returns ogg/bytes, need sf.read
                try:
                    stream, sr = sf.read(BytesIO(chunk))
                    stream = stream.astype(np.float32)
                    if stream.ndim > 1: stream = stream[:, 0]
                    
                    stream = self.resample_audio(stream, sr, self.sample_rate)
                    
                    streamlen = stream.shape[0]
                    idx = 0
                    while streamlen >= self.chunk:
                        eventpoint = {}
                        if first:
                            eventpoint = {'status': 'start', 'text': text}
                            eventpoint.update(**textevent)
                            first = False
                        
                        self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                        streamlen -= self.chunk
                        idx += self.chunk
                except Exception as e:
                    logger.warning(f"Sovits chunk decode error: {e}")

        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)