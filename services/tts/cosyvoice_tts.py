import time
import numpy as np
import resampy
import requests

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg 
        audio_iter = self._fetch_stream(text)
        self._process_stream(audio_iter, msg, sr_orig=24000)

    def _fetch_stream(self, text: str):
        payload = {'tts_text': text, 'prompt_text': self.config.ref_text}
        files = [('prompt_wav', ('prompt_wav', open(self.config.ref_file, 'rb'), 'application/octet-stream'))]
        
        try:
            res = requests.request("GET", f"{self.config.server}/inference_zero_shot", 
                                   data=payload, files=files, stream=True)
            if res.status_code != 200:
                logger.error(f"CosyVoice Error: {res.text}")
                return

            for chunk in res.iter_content(chunk_size=9600): # 24K*20ms*2
                if chunk and self.state == BaseTTS.State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('CosyVoice request error')

    def _process_stream(self, audio_stream, msg: tuple[str, dict], sr_orig: int):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk:
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

        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)