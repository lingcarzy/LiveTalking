import os
import numpy as np
import azure.cognitiveservices.speech as speechsdk

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class AzureTTS(BaseTTS):
    CHUNK_SIZE = 640  # 16kHz, 20ms, 16-bit Mono PCM size
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        self.audio_buffer = b''
        
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        tts_region = os.getenv("AZURE_TTS_REGION")
        speech_endpoint = f"wss://{tts_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=speech_endpoint)
        speech_config.speech_synthesis_voice_name = config.ref_file
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        self.speech_synthesizer.synthesizing.connect(self._on_synthesizing)

    def txt_to_audio(self, msg: tuple[str, dict]):
        msg_text = msg[0]
        # Reset buffer
        self.audio_buffer = b''
        result = self.speech_synthesizer.speak_text_async(msg_text).get()
        
        # Log latency if needed
        # ...

    def _on_synthesizing(self, evt: speechsdk.SpeechSynthesisEventArgs):
        if self.state != BaseTTS.State.RUNNING:
            self.audio_buffer = b''
            return

        self.audio_buffer += evt.result.audio_data
        while len(self.audio_buffer) >= self.CHUNK_SIZE:
            chunk = self.audio_buffer[:self.CHUNK_SIZE]
            self.audio_buffer = self.audio_buffer[self.CHUNK_SIZE:]

            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
            self.parent.put_audio_frame(frame)