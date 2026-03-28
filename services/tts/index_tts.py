import os
import time
import numpy as np
import soundfile as sf
import resampy

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class IndexTTS2(BaseTTS):
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        try:
            from gradio_client import Client, handle_file
            self.client = Client(config.server)
            self.handle_file = handle_file
        except ImportError:
            raise ImportError("IndexTTS2 requires gradio_client: pip install gradio_client")

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        try:
            # Call Gradio API
            result = self.client.predict(
                text=text,
                max_text_tokens_per_segment=120,
                api_name="/on_input_text_change"
            )
            
            segments = []
            if 'value' in result and 'data' in result['value']:
                for item in result['value']['data']:
                    segments.append(item[1]) # text content
            
            if not segments: segments = [text]

            for i, seg_text in enumerate(segments):
                if self.state != BaseTTS.State.RUNNING: break
                
                audio_file = self._generate_segment(seg_text)
                if audio_file:
                    self._process_file(audio_file, msg, is_first=(i==0), is_last=(i==len(segments)-1))
                    
        except Exception as e:
            logger.exception("IndexTTS2 error")

    def _generate_segment(self, text):
        try:
            result = self.client.predict(
                emo_control_method="Same as the voice reference",
                prompt=self.handle_file(self.config.ref_file),
                text=text,
                # ... other params simplified for brevity ...
                api_name="/gen_single"
            )
            return result.get('value') if 'value' in result else None
        except Exception as e:
            logger.error(f"IndexTTS2 generation failed: {e}")
            return None

    def _process_file(self, audio_file, msg, is_first, is_last):
        text, textevent = msg
        try:
            stream, sr = sf.read(audio_file)
            stream = stream.astype(np.float32)
            if stream.ndim > 1: stream = stream[:, 0]
            stream = self.resample_audio(stream, sr, self.sample_rate)
            
            streamlen = stream.shape[0]
            idx = 0
            first_chunk = True
            
            while streamlen >= self.chunk and self.state == BaseTTS.State.RUNNING:
                eventpoint = {}
                if is_first and first_chunk:
                    eventpoint = {'status': 'start', 'text': text}
                    eventpoint.update(**textevent)
                    first_chunk = False
                
                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                streamlen -= self.chunk
            
            if is_last:
                eventpoint = {'status': 'end', 'text': text}
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            
            # Clean up
            if os.path.exists(audio_file): os.remove(audio_file)
        except Exception as e:
            logger.error(f"IndexTTS2 file processing error: {e}")