import requests
import numpy as np
import resampy

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class XTTS(BaseTTS):
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        self.speaker = self._get_speaker()

    def _get_speaker(self):
        try:
            files = {"wav_file": ("reference.wav", open(self.config.ref_file, "rb"))}
            response = requests.post(f"{self.config.server}/clone_speaker", files=files)
            return response.json()
        except Exception as e:
            logger.error(f"XTTS clone speaker failed: {e}")
            return {}

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.speaker["text"] = text
        self.speaker["language"] = "zh-cn"
        self.speaker["stream_chunk_size"] = "20"

        try:
            res = requests.post(f"{self.config.server}/tts_stream", json=self.speaker, stream=True)
            first = True
            last_stream = np.array([], dtype=np.float32)

            for chunk in res.iter_content(chunk_size=None):
                if chunk:
                    stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                    stream = self.resample_audio(stream, 24000, self.sample_rate)
                    stream = np.concatenate((last_stream, stream))

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
                    last_stream = stream[idx:]
            
            eventpoint = {'status': 'end', 'text': text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
        except Exception as e:
            logger.exception("XTTS error")