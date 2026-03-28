import time
import numpy as np
import resampy
import requests
from typing import Iterator

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class FishTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]): 
        text, textevent = msg
        audio_iter = self._fetch_stream(text)
        self._process_stream(audio_iter, msg, sr_orig=44100)

    def _fetch_stream(self, text: str) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'reference_id': self.config.ref_file,
            'format': 'wav',
            'streaming': True,
            'use_memory_cache': 'on'
        }
        
        try:
            res = requests.post(
                f"{self.config.server}/v1/tts",
                json=req,
                stream=True,
                headers={"content-type": "application/json"},
            )
            
            if res.status_code != 200:
                logger.error(f"FishTTS Error: {res.text}")
                return

            for chunk in res.iter_content(chunk_size=17640): # 44100*20ms*2
                if chunk and self.state == BaseTTS.State.RUNNING:
                    yield chunk
                    
        except Exception as e:
            logger.exception('FishTTS request error')

    def _process_stream(self, audio_stream: Iterator[bytes], msg: tuple[str, dict], sr_orig: int):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk:
                # int16 -> float32
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = self.resample_audio(stream, sr_orig, self.sample_rate)
                
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
        
        # End event
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)