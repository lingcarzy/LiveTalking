###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import torch
import numpy as np

import cv2

from avatars.audio_features.mel import MelASR
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from avatars.avatar_utils import get_avatar_path, get_inference_device, get_mirror_batch_indices, load_pickle_file, load_sorted_images, load_torch_file, warm_up_avatar_model
from utils.logger import logger
from registry import register

device = get_inference_device()

def _load(checkpoint_path):
    if device == 'cuda':
        return load_torch_file(checkpoint_path)
    return load_torch_file(checkpoint_path, map_location=lambda storage, loc: storage)

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = get_avatar_path(avatar_id)
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"

    coord_list_cycle = load_pickle_file(coords_path)
    frame_list_cycle = load_sorted_images(full_imgs_path)
    face_list_cycle = load_sorted_images(face_imgs_path)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    def build_inputs():
        return (
            torch.ones(batch_size, 1, 80, 16).to(device),
            torch.ones(batch_size, 6, modelres, modelres).to(device),
        )

    warm_up_avatar_model(build_inputs, model)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model
        self._compose_buf = None

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        batch_indices = get_mirror_batch_indices(length, index, self.batch_size)
        img_batch = [self.face_list_cycle[idx] for idx in batch_indices]
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)
        face = img_batch[0]

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        source_frame = self.frame_list_cycle[idx]
        if self._compose_buf is None or self._compose_buf.shape != source_frame.shape:
            self._compose_buf = np.empty_like(source_frame)
        np.copyto(self._compose_buf, source_frame)
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        self._compose_buf[y1:y2, x1:x2] = res_frame
        return self._compose_buf

