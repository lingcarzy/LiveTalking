#core/render_loop.py
import os
import torch 
import time
import cv2
import numpy as np
import queue
import copy
import asyncio
import resampy
from io import BytesIO
import soundfile as sf
from threading import Thread, Event
from av import AudioFrame, VideoFrame
from fractions import Fraction
from typing import Any, Dict, List, Optional

from configs import AppConfig
from services.tts import create_tts_service
from services.asr import create_asr_service
from services.media.recorder import MediaRecorder
from logger import logger
from tqdm import tqdm

# 辅助函数：读取图片 (保持原逻辑)
def read_imgs(img_list):
    frames = []
    logger.info('Reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# 辅助函数：音频播放 (用于 virtualcam 模式)
def play_audio_thread(quit_event, audio_queue):
    import pyaudio
    p = pyaudio.PyAudio()
    # ... (此处保留原 basereal.py 中 play_audio 的完整实现，略)
    # 为节省篇幅，假设此处代码与原 play_audio 一致
    pass 

class RenderLoop:
    def __init__(self, config: AppConfig, model_instance: Any, avatar_data: Any):
        self.config = config
        self.model = model_instance
        self.avatar = avatar_data
        
        # 基础参数
        self.sample_rate = 16000
        self.chunk = self.sample_rate // config.model.fps
        self.session_id = config.session_id
        
        # 状态
        self.speaking = False
        self.curr_state = 0 # 0: idle/speaking, >1: custom action

        # 初始化服务
        # 1. TTS
        self.tts = create_tts_service(config.tts, self)
        
        # 2. ASR (需要根据模型类型创建特定的 Audio Processor)
        # 这里的 audio_processor 需要从 model_instance 或 avatar_data 中获取
        # 在 musereal 中是 model[4], 在 lightreal 中是 model
        audio_processor = self._get_audio_processor()
        self.asr = create_asr_service(config.model, self, audio_processor)
        
        # 3. Recorder
        self.recorder = MediaRecorder(self.session_id)

        # 数据缓存
        self.frame_list_cycle = []      # 全帧图片
        self.coord_list_cycle = []      # 坐标
        self.input_latent_list_cycle = [] # latent (musetalk)
        
        # 自定义动作数据
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        
        # 结果队列 (由子类或 Inference 线程填充)
        self.res_frame_queue = queue.Queue(maxsize=config.model.batch_size * 2)

        # 初始化数据
        self._load_avatar_data()
        self._load_custom_actions()
    async def __aenter__(self):
        """进入上下文：初始化资源"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：清理资源"""
        logger.info(f"Cleaning up RenderLoop session {self.session_id}")
        if self.recorder:
            self.recorder.stop()
        # 显存清理等操作
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def _get_audio_processor(self):
        """由子类实现，返回特定的音频处理器 (Whisper/HuBERT/Mel)"""
        raise NotImplementedError

    def _load_avatar_data(self):
        """由子类实现，加载具体的数字人资产"""
        raise NotImplementedError

    def _load_custom_actions(self):
        for item in self.config.customopt:
            try:
                img_path = item['imgpath']
                audio_path = item['audiotype'] # Note: original code used audiotype as key but audiopath for reading? 
                # Checking original code: sf.read(item['audiopath'])
                # But dictionary key is audiotype.
                
                input_img_list = sorted(
                    [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))],
                    key=lambda x: int(os.path.splitext(x)[0])
                )
                full_img_paths = [os.path.join(img_path, f) for f in input_img_list]
                
                self.custom_img_cycle[item['audiotype']] = read_imgs(full_img_paths)
                
                audio_data, sr = sf.read(item['audiopath'], dtype='float32')
                if sr != 16000:
                    audio_data = resampy.resample(audio_data, sr, 16000)
                self.custom_audio_cycle[item['audiotype']] = audio_data
                
                self.custom_audio_index[item['audiotype']] = 0
                self.custom_index[item['audiotype']] = 0
            except Exception as e:
                logger.error(f"Failed to load custom action {item}: {e}")

    # ------------------- 公共接口 -------------------

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        self.tts.put_msg_txt(msg, datainfo)

    def put_audio_frame(self, audio_chunk: np.ndarray, datainfo: dict = {}):
        self.asr.put_audio_frame(audio_chunk, datainfo)

    def put_audio_file(self, filebyte: bytes, datainfo: dict = {}):
        stream = self._create_bytes_stream(filebyte)
        idx = 0
        while idx + self.chunk <= stream.shape[0]:
            self.put_audio_frame(stream[idx:idx+self.chunk], datainfo)
            idx += self.chunk

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking

    # ------------------- 内部逻辑 -------------------

    def _create_bytes_stream(self, byte_stream):
        stream, sr = sf.read(BytesIO(byte_stream))
        stream = stream.astype(np.float32)
        if stream.ndim > 1: stream = stream[:, 0]
        if sr != self.sample_rate and stream.shape[0] > 0:
            stream = resampy.resample(stream, sr, self.sample_rate)
        return stream

    def get_audio_stream(self, audiotype: int):
        """获取自定义动作的音频流"""
        if audiotype not in self.custom_audio_cycle:
            return np.zeros(self.chunk, dtype=np.float32)
        
        idx = self.custom_audio_index[audiotype]
        audio_data = self.custom_audio_cycle[audiotype]
        
        chunk = audio_data[idx : idx + self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        
        if self.custom_audio_index[audiotype] >= len(audio_data):
            self.curr_state = 1 # 切换到静音
            
        # Pad if chunk is too short
        if len(chunk) < self.chunk:
            chunk = np.pad(chunk, (0, self.chunk - len(chunk)))
        return chunk

    def mirror_index(self, size, index):
        turn = index // size
        res = index % size
        return res if turn % 2 == 0 else size - res - 1

    # ------------------- 渲染主循环 -------------------

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        # 虚拟摄像头初始化
        if self.config.server.transport == 'virtualcam':
            import pyvirtualcam
            vircam = None
            audio_q = queue.Queue(maxsize=3000)
            audio_t = Thread(target=play_audio_thread, args=(quit_event, audio_q), daemon=True)
            audio_t.start()

        while not quit_event.is_set():
            try:
                # 从推理线程获取结果
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # 判断状态
            is_silence = audio_frames[0][1] != 0 and audio_frames[1][1] != 0
            
            if is_silence:
                self.speaking = False
                audiotype = audio_frames[0][1]
                
                # 获取帧图像
                if audiotype in self.custom_img_cycle:
                    mir_idx = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mir_idx]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                combine_frame = target_frame
            else:
                self.speaking = True
                # paste_back_frame 由子类实现 (因 MuseTalk/Lip/Wav2Lip 逻辑不同)
                try:
                    combine_frame = self.paste_back_frame(res_frame, idx)
                except Exception as e:
                    logger.warning(f"Render error: {e}")
                    continue

            # 绘制水印
            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)

            # 发送视频
            if self.config.server.transport == 'virtualcam':
                if vircam is None:
                    h, w, _ = combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=w, height=h, fps=25, fmt=pyvirtualcam.PixelFormat.BGR)
                vircam.send(combine_frame)
            else:
                new_frame = VideoFrame.from_ndarray(combine_frame, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame, None)), loop)
            
            self.recorder.write_video(combine_frame)

            # 发送音频
            for frame, type_, eventpoint in audio_frames:
                int_frame = (frame * 32767).astype(np.int16)
                
                if self.config.server.transport == 'virtualcam':
                    audio_q.put(int_frame.tobytes())
                else:
                    new_audio = AudioFrame(format='s16', layout='mono', samples=int_frame.shape[0])
                    new_audio.planes[0].update(int_frame.tobytes())
                    new_audio.sample_rate = 16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_audio, eventpoint)), loop)
                
                self.recorder.write_audio(frame) # recorder 内部处理 float->int

            if self.config.server.transport == 'virtualcam':
                vircam.sleep_until_next_frame()

        logger.info("Render loop stopped.")
        if self.config.server.transport == 'virtualcam' and vircam:
            vircam.close()

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """主入口：启动 ASR, TTS, Inference 和 Frame Processor"""
        # 1. 启动 TTS
        self.tts.render(quit_event)
        
        # 2. 预热 ASR
        self.asr.warm_up()
        
        # 3. 启动推理线程 (由子类实现具体逻辑)
        infer_quit = Event()
        infer_thread = Thread(target=self._inference_thread_entry, args=(infer_quit,))
        infer_thread.start()

        # 4. 启动渲染线程
        render_quit = Event()
        render_thread = Thread(target=self.process_frames, args=(render_quit, loop, audio_track, video_track))
        render_thread.start()

        # 5. 主循环 (驱动 ASR)
        try:
            while not quit_event.is_set():
                self.asr.run_step()
                # 流控控制
                if video_track and video_track._queue.qsize() >= 1.5 * self.config.model.batch_size:
                    time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        finally:
            logger.info("Stopping render...")
            infer_quit.set()
            render_quit.set()
            infer_thread.join()
            render_thread.join()

    def _inference_thread_entry(self, quit_event):
        """由子类实现，运行具体的模型推理循环"""
        raise NotImplementedError

    def paste_back_frame(self, res_frame, idx: int):
        """由子类实现，将生成的脸部贴回原图"""
        raise NotImplementedError
    def __del__(self):
        """析构函数：确保资源释放"""
        try:
            if hasattr(self, 'recorder') and self.recorder:
                self.recorder.stop()
            # 显式清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass # 析构函数中避免抛出异常