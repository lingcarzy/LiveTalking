import subprocess
import os
import numpy as np
from logger import logger

class MediaRecorder:
    def __init__(self, session_id: int, width: int = 0, height: int = 0):
        self.session_id = session_id
        self.width = width
        self.height = height
        
        self.is_recording = False
        self._video_pipe = None
        self._audio_pipe = None

    def set_resolution(self, width: int, height: int):
        self.width = width
        self.height = height

    def start(self):
        if self.is_recording:
            return
        
        if self.width == 0 or self.height == 0:
            logger.error("Cannot start recording without resolution info.")
            return

        logger.info(f"Starting recording for session {self.session_id} ({self.width}x{self.height})...")
        
        # 视频管道
        v_cmd = [
            'ffmpeg', '-y', '-an',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}",
            '-r', '25',
            '-i', '-',
            '-pix_fmt', 'yuv420p', '-vcodec', 'h264',
            f'temp_{self.session_id}.mp4'
        ]
        self._video_pipe = subprocess.Popen(v_cmd, shell=False, stdin=subprocess.PIPE)

        # 音频管道
        a_cmd = [
            'ffmpeg', '-y', '-vn',
            '-f', 's16le', '-ac', '1', '-ar', '16000',
            '-i', '-',
            '-acodec', 'aac',
            f'temp_{self.session_id}.aac'
        ]
        self._audio_pipe = subprocess.Popen(a_cmd, shell=False, stdin=subprocess.PIPE)
        
        self.is_recording = True

    def write_video(self, frame: np.ndarray):
        if self.is_recording and self._video_pipe:
            try:
                self._video_pipe.stdin.write(frame.tobytes())
            except BrokenPipeError:
                logger.error("Video recording pipe broken.")
                self.stop() # 自动停止

    def write_audio(self, frame: np.ndarray):
        if self.is_recording and self._audio_pipe:
            try:
                # frame 是 float32，需要转换为 int16
                audio_data = (frame * 32767).astype(np.int16)
                self._audio_pipe.stdin.write(audio_data.tobytes())
            except BrokenPipeError:
                logger.error("Audio recording pipe broken.")
                self.stop()

    def stop(self):
        if not self.is_recording:
            return
        
        logger.info(f"Stopping recording for session {self.session_id}...")
        self.is_recording = False
        
        if self._video_pipe:
            self._video_pipe.stdin.close()
            self._video_pipe.wait()
            
        if self._audio_pipe:
            self._audio_pipe.stdin.close()
            self._audio_pipe.wait()

        # 合并音视频
        output_path = f"data/record_{self.session_id}.mp4"
        cmd_combine = (
            f"ffmpeg -y -i temp_{self.session_id}.aac -i temp_{self.session_id}.mp4 "
            f"-c:v copy -c:a copy {output_path}"
        )
        os.system(cmd_combine)
        logger.info(f"Recording saved to {output_path}")

    def __del__(self):
        if self.is_recording:
            self.stop()