import os
import time
import hmac
import hashlib
import base64
import json
import uuid
import numpy as np
import requests

from .base_tts import BaseTTS
from configs import TTSConfig
from logger import logger

_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"

class TencentTTS(BaseTTS):
    def __init__(self, config: TTSConfig, parent_ref):
        super().__init__(config, parent_ref)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(config.ref_file) # VoiceType is int
        self.codec = "pcm"
        self.sample_rate = 16000

    def _gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        return base64.b64encode(hmacstr).decode('utf-8')

    def _gen_params(self, session_id, text):
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = 0
        params['Volume'] = 0
        params['SessionId'] = session_id
        params['Text'] = text
        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        session_id = str(uuid.uuid1())
        params = self._gen_params(session_id, text)
        signature = self._gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(params), stream=True)
            first = True
            last_stream = np.array([], dtype=np.float32)

            for chunk in res.iter_content(chunk_size=6400): # 16K*20ms*2
                if first:
                    try:
                        # Check for error response
                        rsp = json.loads(chunk)
                        logger.error(f"Tencent TTS error: {rsp.get('Response', {}).get('Error', {}).get('Message')}")
                        return
                    except:
                        first = False
                
                if chunk and self.state == BaseTTS.State.RUNNING:
                    stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                    stream = np.concatenate((last_stream, stream))
                    
                    streamlen = stream.shape[0]
                    idx = 0
                    while streamlen >= self.chunk:
                        eventpoint = {'status': 'start' if first else 'middle'} # Simplified status
                        self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                        streamlen -= self.chunk
                        idx += self.chunk
                    last_stream = stream[idx:]
            
            # End event
            eventpoint = {'status': 'end', 'text': text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

        except Exception as e:
            logger.exception('TencentTTS error')