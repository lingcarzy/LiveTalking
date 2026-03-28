import torch
import numpy as np
import queue
import time
import copy
import os
import pickle
import glob
import cv2
from threading import Event, Thread

from core.render_loop import RenderLoop
from logger import logger
from tqdm import tqdm

# Helper to read images, can be moved to utils if needed
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class MuseRenderLoop(RenderLoop):
    def _get_audio_processor(self):
        # model_instance 传入的是 (vae, unet, pe, timesteps, audio_processor)
        return self.model[4] 

    def _load_avatar_data(self):
        # avatar_data 传入的是 load_avatar 的返回值
        (self.frame_list_cycle, self.mask_list_cycle, 
         self.coord_list_cycle, self.mask_coords_list_cycle, 
         self.input_latent_list_cycle) = self.avatar

    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2-x1, y2-y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]

        # 使用项目原有的混合逻辑
        from musetalk.myutil import get_image_blending
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        return combine_frame

    def _inference_thread_entry(self, quit_event):
        # 解构模型组件
        vae, unet, pe, timesteps, _ = self.model
        length = len(self.input_latent_list_cycle)
        index = 0
        count = 0
        counttime = 0
        
        logger.info('MuseRenderLoop inference thread started')
        
        while not quit_event.is_set():
            starttime = time.perf_counter()
            
            # 1. 获取特征
            try:
                whisper_chunks = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 2. 获取音频帧用于同步
            is_all_silence = True
            audio_frames = []
            for _ in range(self.config.model.batch_size * 2):
                frame, type_, eventpoint = self.asr.output_queue.get()
                audio_frames.append((frame, type_, eventpoint))
                if type_ == 0: is_all_silence = False
            
            # 3. 推理或静音处理
            if is_all_silence:
                for i in range(self.config.model.batch_size):
                    self.res_frame_queue.put((None, self.mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1
            else:
                t = time.perf_counter()
                
                # 数据准备
                whisper_batch = np.stack(whisper_chunks)
                latent_batch = []
                for i in range(self.config.model.batch_size):
                    idx = self.mirror_index(length, index + i)
                    latent_batch.append(self.input_latent_list_cycle[idx])
                latent_batch = torch.cat(latent_batch, dim=0)

                audio_feature_batch = torch.from_numpy(whisper_batch).to(
                    device=unet.device, dtype=unet.model.dtype
                )
                audio_feature_batch = pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                # 推理
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)

                # 性能统计
                counttime += (time.perf_counter() - t)
                count += self.config.model.batch_size
                if count >= 100:
                    logger.info(f"Muse Inference FPS: {count/counttime:.4f}")
                    count = 0
                    counttime = 0

                # 推送结果
                for i, res_frame in enumerate(recon):
                    self.res_frame_queue.put((res_frame, self.mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1