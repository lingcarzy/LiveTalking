import torch
import numpy as np
import queue
import time
import copy
import os
import cv2
import pickle
import glob
from threading import Event, Thread

from core.render_loop import RenderLoop
from logger import logger
from tqdm import tqdm

# 引入 Wav2Lip 依赖
try:
    from wav2lip.models import Wav2Lip
except ImportError:
    Wav2Lip = None

# Helper
def read_imgs(img_list):
    frames = []
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class LipRenderLoop(RenderLoop):
    def _get_audio_processor(self):
        # LipASR 不需要 audio_processor，这里返回 None
        return None

    def _load_avatar_data(self):
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = self.avatar

    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2-x1, y2-y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame

    def _inference_thread_entry(self, quit_event):
        model = self.model
        length = len(self.face_list_cycle)
        index = 0
        count = 0
        counttime = 0
        device = next(model.parameters()).device

        logger.info('LipRenderLoop inference thread started')

        while not quit_event.is_set():
            try:
                mel_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            is_all_silence = True
            audio_frames = []
            for _ in range(self.config.model.batch_size * 2):
                frame, type_, eventpoint = self.asr.output_queue.get()
                audio_frames.append((frame, type_, eventpoint))
                if type_ == 0: is_all_silence = False

            if is_all_silence:
                for i in range(self.config.model.batch_size):
                    self.res_frame_queue.put((None, self.mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1
            else:
                t = time.perf_counter()
                
                # 准备数据
                img_batch = []
                for i in range(self.config.model.batch_size):
                    idx = self.mirror_index(length, index + i)
                    face = self.face_list_cycle[idx]
                    img_batch.append(face)
                
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                
                img_masked = img_batch.copy()
                img_masked[:, face.shape[0]//2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                # 推理
                with torch.no_grad():
                    pred = model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                counttime += (time.perf_counter() - t)
                count += self.config.model.batch_size
                if count >= 100:
                    logger.info(f"Lip Inference FPS: {count/counttime:.4f}")
                    count = 0
                    counttime = 0

                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, self.mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1