# services/tts/cosyvoice3_tts.py
import time
import numpy as np
import requests

from .base_tts import BaseTTS, State
from configs import TTSConfig
from logger import logger


class CosyVoice3TTS(BaseTTS):
    """
    CosyVoice3 FastAPI 流式 TTS 适配器
    适配 cosyvoice3_livetalking_api.py 服务
    """
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        instruct = textevent.get('instruct', None)
        audio_iter = self._fetch_stream(text, instruct)
        self._process_stream(audio_iter, msg, sr_orig=24000)

    def _fetch_stream(self, text: str, instruct: str = None):
        start = time.perf_counter()
        params = {'text': text}
        if instruct:
            params['instruct'] = instruct
        
        try:
            res = requests.get(
                f"{self.config.server}/tts",
                params=params,
                stream=True,
                timeout=30
            )
            
            req_time = time.perf_counter()
            logger.info(f"CosyVoice3 API 请求耗时: {req_time-start:.3f}s")

            if res.status_code != 200:
                logger.error(f"CosyVoice3 API 错误: {res.status_code} - {res.text}")
                return
            
            first = True
            chunk_count = 0
            
            for chunk in res.iter_content(chunk_size=9600):
                if first:
                    first_chunk_time = time.perf_counter()
                    logger.info(f"CosyVoice3 首包延迟: {first_chunk_time-start:.3f}s")
                    first = False
                
                if chunk and self.state == State.RUNNING:
                    chunk_count += 1
                    yield chunk
            
            total_time = time.perf_counter()
            logger.info(f"CyVoice3 总接收时间: {total_time-start:.3f}s, 共 {chunk_count} 个包")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"CosyVoice3 API 连接失败: {self.config.server} - {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"CosyVoice3 API 请求超时: {e}")
        except Exception as e:
            logger.exception(f"CosyVoice3 API 调用异常: {e}")

    def _process_stream(self, audio_stream, msg: tuple[str, dict], sr_orig: int):
        text, textevent = msg
        first = True
        
        try:
            for chunk in audio_stream:
                if chunk and len(chunk) > 0:
                    stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                    
                    if sr_orig != self.sample_rate:
                        stream = self.resample_audio(stream, sr_orig, self.sample_rate)
                    
                    streamlen = stream.shape[0]
                    idx = 0
                    
                    while streamlen >= self.chunk:
                        if self.state != State.RUNNING:
                            return
                        
                        eventpoint = {}
                        if first:
                            eventpoint = {'status': 'start', 'text': text}
                            eventpoint.update(**textevent)
                            first = False
                        
                        self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                        streamlen -= self.chunk
                        idx += self.chunk
            
            # 4. 修改这里：直接使用 State
            if self.state == State.RUNNING:
                eventpoint = {'status': 'end', 'text': text}
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
                
        except Exception as e:
            logger.exception(f"CosyVoice3 流处理异常: {e}")
            eventpoint = {'status': 'end', 'text': text, 'error': str(e)}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)