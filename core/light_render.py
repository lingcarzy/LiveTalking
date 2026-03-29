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

from core.utils import read_imgs
from core.render_loop import RenderLoop
from logger import logger
from tqdm import tqdm


class LightRenderLoop(RenderLoop):
    def _get_audio_processor(self):
        # model_instance 是 audio_processor
        return self.model

    def _load_avatar_data(self):
        # avatar_data 包含 (model, frames, faces, coords)
        # 这里的 model 实际上已经在 RenderLoop 初始化时通过 _get_audio_processor 分离出去了
        # 所以这里只存其他数据
        # 但注意，LightReal 原逻辑中 avatar 包含 model，这里我们需要特殊处理
        # 实际上 init 中 self.model 已经是 audio_processor 了
        self.frame_list_cycle = self.avatar[1]
        self.face_list_cycle = self.avatar[2]
        self.coord_list_cycle = self.avatar[3]
        # self.avatar[0] 是 ultralight model, 但推理逻辑需要它
        self.inference_model = self.avatar[0]

    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = self.frame_list_cycle[idx].copy()
        x1, y1, x2, y2 = bbox

        crop_img = self.face_list_cycle[idx]
        crop_img_ori = crop_img.copy()
        
        crop_img_ori[4:164, 4:164] = pred_frame.astype(np.uint8)
        crop_img_ori = cv2.resize(crop_img_ori, (x2-x1, y2-y1))
        combine_frame[y1:y2, x1:x2] = crop_img_ori
        return combine_frame

    def _inference_thread_entry(self, quit_event):
        model = self.inference_model
        length = len(self.face_list_cycle)
        index = 0
        count = 0
        counttime = 0
        device = next(model.parameters()).device

        logger.info('LightRenderLoop inference thread started')

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
                
                img_batch = []
                for i in range(self.config.model.batch_size):
                    idx = self.mirror_index(length, index + i)
                    crop_img = self.face_list_cycle[idx]
                    img_real_ex = crop_img[4:164, 4:164].copy()
                    img_masked = cv2.rectangle(img_real_ex.copy(), (5,5,150,145), (0,0,0), -1)

                    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
                    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)

                    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
                    img_masked_T = torch.from_numpy(img_masked / 255.0)
                    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
                    img_batch.append(img_concat_T)

                # 音频特征处理
                reshaped_mel_batch = [arr.reshape(16, 32, 32) for arr in mel_batch]
                mel_batch_tensor = torch.stack([torch.from_numpy(arr) for arr in reshaped_mel_batch])
                img_batch_tensor = torch.stack(img_batch).squeeze(1)

                # 推理
                with torch.no_grad():
                    pred = model(img_batch_tensor.cuda(), mel_batch_tensor.cuda())
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                counttime += (time.perf_counter() - t)
                count += self.config.model.batch_size
                if count >= 100:
                    logger.info(f"Light Inference FPS: {count/counttime:.4f}")
                    count = 0
                    counttime = 0

                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, self.mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1