import os
import json
import uuid
import copy
import gzip
import asyncio
import numpy as np
import websockets

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

class DoubaoTTS(BaseTTS):
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        self.appid = os.getenv("DOUBAO_APPID")
        self.token = os.getenv("DOUBAO_TOKEN")
        self.api_url = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
        
        self.request_json = {
            "app": {"appid": self.appid, "token": "access_token", "cluster": "volcano_tts"},
            "user": {"uid": "xxx"},
            "audio": {
                "voice_type": "xxx", "encoding": "pcm", "rate": 16000,
                "speed_ratio": 1.0, "volume_ratio": 1.0, "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": "xxx", "text": "text", "text_type": "plain", "operation": "xxx"
            }
        }

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        asyncio.new_event_loop().run_until_complete(self._run_ws(text, textevent))

    async def _run_ws(self, text, textevent):
        voice_type = self.config.ref_file
        submit_req = copy.deepcopy(self.request_json)
        submit_req["user"]["uid"] = str(self.parent.sessionid) # Assuming parent has sessionid
        submit_req["audio"]["voice_type"] = voice_type
        submit_req["request"]["text"] = text
        submit_req["request"]["reqid"] = str(uuid.uuid4())
        submit_req["request"]["operation"] = "submit"
        
        payload_bytes = gzip.compress(str.encode(json.dumps(submit_req)))
        default_header = bytearray(b'\x11\x10\x11\x00')
        full_client_request = bytearray(default_header)
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        full_client_request.extend(payload_bytes)

        header = {"Authorization": f"Bearer; {self.token}"}
        first = True
        last_stream = np.array([], dtype=np.float32)

        try:
            async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
                await ws.send(full_client_request)
                while True:
                    res = await ws.recv()
                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0: continue
                        
                        sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                        audio_payload = payload[8:]

                        if audio_payload and self.state == BaseTTS.State.RUNNING:
                            stream = np.frombuffer(audio_payload, dtype=np.int16).astype(np.float32) / 32767
                            stream = np.concatenate((last_stream, stream))
                            
                            streamlen = stream.shape[0]
                            idx = 0
                            while streamlen >= self.chunk:
                                eventpoint = {'status': 'start', 'text': text} if first else {}
                                if first: eventpoint.update(**textevent); first = False
                                
                                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                                streamlen -= self.chunk
                                idx += self.chunk
                            last_stream = stream[idx:]

                        if sequence_number < 0: break
                    else: break
        except Exception as e:
            logger.exception("DoubaoTTS error")
        
        # End event
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)