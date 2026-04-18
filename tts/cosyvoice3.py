import time
import struct
from typing import Iterator, Optional

import numpy as np
import requests
import resampy
from requests.exceptions import ChunkedEncodingError, RequestException

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register


@register("tts", "cosyvoice3")
class CosyVoice3TTS(BaseTTS):
    """CosyVoice3 FastAPI adapter.

    Compatible with API shape:
    POST /tts
    body: {"text": str, "instruct": str|None, "speed": float}
    response: audio/wav (streaming) or raw pcm stream.
    """

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        instruct = textevent.get("tts", {}).get("ref_text", self.opt.REF_TEXT)
        speed = textevent.get("tts", {}).get("speed", 1.0)

        self.stream_tts(
            self.cosy_voice3(
                text=text,
                instruct=instruct,
                speed=speed,
                server_url=self.opt.TTS_SERVER,
            ),
            msg,
        )

    def cosy_voice3(self, text: str, instruct: Optional[str], speed: float, server_url: str) -> Iterator[bytes]:
        start = time.perf_counter()

        payload = {
            "text": text,
            "instruct": instruct,
            "speed": float(speed),
        }

        max_first_chunk_retries = 1
        attempt = 0
        while attempt <= max_first_chunk_retries:
            attempt += 1
            emitted_any = False
            first = True
            try:
                with requests.post(
                    f"{server_url.rstrip('/')}/tts",
                    json=payload,
                    stream=True,
                    timeout=30,
                ) as res:
                    end = time.perf_counter()
                    logger.info(f"cosyvoice3 Time to make POST(attempt={attempt}): {end - start}s")

                    if res.status_code != 200:
                        logger.error("cosyvoice3 error(%s): %s", res.status_code, res.text)
                        return

                    source_sample_rate = self._resolve_source_sample_rate(res)
                    header_buffer = bytearray()
                    header_done = False

                    try:
                        for chunk in res.iter_content(chunk_size=9600):
                            if not chunk or self.state != State.RUNNING:
                                continue

                            if not header_done:
                                header_buffer.extend(chunk)
                                parsed = self._try_strip_wav_header(header_buffer)
                                if parsed is None:
                                    # Need more bytes to fully parse wav header.
                                    continue

                                wav_sample_rate, payload_bytes = parsed
                                if wav_sample_rate is not None:
                                    source_sample_rate = wav_sample_rate
                                self._source_sample_rate = source_sample_rate
                                header_done = True

                                if first:
                                    end = time.perf_counter()
                                    logger.info(f"cosyvoice3 Time to first chunk: {end - start}s")
                                    first = False

                                if payload_bytes:
                                    emitted_any = True
                                    yield payload_bytes
                                continue

                            if first:
                                end = time.perf_counter()
                                logger.info(f"cosyvoice3 Time to first chunk: {end - start}s")
                                first = False

                            emitted_any = True
                            yield chunk
                    except ChunkedEncodingError as e:
                        if emitted_any:
                            logger.warning("cosyvoice3 stream ended prematurely, treat as EOS: %s", e)
                        elif attempt <= max_first_chunk_retries:
                            logger.warning("cosyvoice3 stream closed before first chunk, retrying(%s/%s): %s",
                                           attempt, max_first_chunk_retries, e)
                            continue
                        else:
                            logger.warning("cosyvoice3 stream closed before first chunk after retries, skip this utterance: %s", e)
                        
                    logger.info("cosyvoice3 stream end, source_sample_rate=%s", self._source_sample_rate)
                    return
            except RequestException as e:
                if attempt <= max_first_chunk_retries:
                    logger.warning("cosyvoice3 request failed before first chunk, retrying(%s/%s): %s",
                                   attempt, max_first_chunk_retries, e)
                    continue
                logger.warning("cosyvoice3 request failed after retries: %s", e)
                return
            except Exception:
                logger.exception("cosyvoice3")
                return

    def stream_tts(self, audio_stream: Iterator[bytes], msg: tuple[str, dict]):
        text, textevent = msg
        first = True

        byte_remainder = b""
        sample_remainder = np.array([], dtype=np.float32)

        for chunk in audio_stream:
            if not chunk:
                continue

            pcm_bytes = byte_remainder + chunk
            if len(pcm_bytes) % 2 == 1:
                byte_remainder = pcm_bytes[-1:]
                pcm_bytes = pcm_bytes[:-1]
            else:
                byte_remainder = b""

            if not pcm_bytes:
                continue

            stream = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

            sr_orig = self._source_sample_rate if self._source_sample_rate > 0 else 24000
            if sr_orig != self.sample_rate and stream.shape[0] > 0:
                stream = resampy.resample(x=stream, sr_orig=sr_orig, sr_new=self.sample_rate)

            if sample_remainder.shape[0] > 0:
                stream = np.concatenate([sample_remainder, stream])

            streamlen = stream.shape[0]
            idx = 0
            while streamlen - idx >= self.chunk and self.state == State.RUNNING:
                eventpoint = {}
                if first:
                    eventpoint = {"status": "start", "text": text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk

            sample_remainder = stream[idx:] if idx < streamlen else np.array([], dtype=np.float32)

        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def _resolve_source_sample_rate(self, res: requests.Response) -> int:
        """Prefer explicit header from upstream, then fallback to common defaults."""
        sr_header = res.headers.get("X-Sample-Rate")
        if sr_header:
            try:
                return int(sr_header)
            except ValueError:
                logger.warning("Invalid X-Sample-Rate header: %s", sr_header)
        # CosyVoice3 default sample rate is often 22050, while some deployments use 24000.
        return 22050

    def _try_strip_wav_header(self, buffer: bytearray) -> Optional[tuple[Optional[int], bytes]]:
        """Try parsing a wav header from accumulated bytes.

        Returns:
            None: need more bytes.
            (sample_rate, payload_bytes): parsed successfully.
                sample_rate is None when content is not wav and payload is raw bytes.
        """
        if len(buffer) < 4:
            return None

        # Not wav, treat current bytes as raw pcm.
        if buffer[:4] != b"RIFF":
            payload = bytes(buffer)
            buffer.clear()
            return None, payload

        if len(buffer) < 12:
            return None
        if buffer[8:12] != b"WAVE":
            payload = bytes(buffer)
            buffer.clear()
            return None, payload

        pos = 12
        sample_rate = None
        data_offset = None

        while True:
            if len(buffer) < pos + 8:
                return None

            chunk_id = bytes(buffer[pos:pos + 4])
            chunk_size = struct.unpack_from("<I", buffer, pos + 4)[0]
            chunk_start = pos + 8
            if chunk_id == b"data":
                # For streaming WAV, data chunk size may be 0xFFFFFFFF (unknown).
                # Do not wait for the whole chunk; everything after chunk header is payload.
                data_offset = chunk_start
                break

            chunk_end = chunk_start + chunk_size
            if chunk_size % 2 == 1:
                chunk_end += 1

            if len(buffer) < chunk_end:
                return None

            if chunk_id == b"fmt " and chunk_size >= 16:
                # fmt chunk: AudioFormat(2), Channels(2), SampleRate(4), ...
                sample_rate = struct.unpack_from("<I", buffer, chunk_start + 4)[0]

            pos = chunk_end

        payload = bytes(buffer[data_offset:])
        buffer.clear()
        return sample_rate, payload

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self._source_sample_rate = 22050
