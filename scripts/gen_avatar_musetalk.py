# scripts/gen_avatar_musetalk.py
"""
离线工具：生成 MuseTalk 所需的数字人资产
用法: python -m scripts.gen_avatar_musetalk --file <视频或图片路径> --avatar_id <名称>
"""
import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 注意：这里的导入路径可能需要根据你实际的项目结构调整
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.utils import load_all_model

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from musetalk.utils.face_parsing import FaceParsing


def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    return os.path.splitext(file_path)[1].lower() in video_exts

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def create_musetalk_human(file, avatar_id, args, vae, fp):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 指向项目根目录
    save_path = os.path.join(current_dir, f'./data/avatars/{avatar_id}')
    save_full_path = os.path.join(save_path, 'full_imgs')
    mask_out_path = os.path.join(save_path, 'mask')
    create_dir(save_path)
    create_dir(save_full_path)
    create_dir(mask_out_path)

    coords_path = os.path.join(save_path, 'coords.pkl')
    latents_out_path = os.path.join(save_path, 'latents.pt')
    mask_coords_path = os.path.join(save_path, 'mask_coords.pkl')

    with open(os.path.join(save_path, 'avator_info.json'), "w") as f:
        json.dump({"avatar_id": avatar_id, "video_path": file, "bbox_shift": args.bbox_shift}, f)

    # 1. 提取图片帧
    if os.path.isfile(file):
        if is_video_file(file):
            video2imgs(file, save_full_path)
        else:
            shutil.copyfile(file, f"{save_full_path}/{os.path.basename(file)}")
    else:
        files = sorted([f for f in os.listdir(file) if f.endswith(".png")])
        for filename in files:
            shutil.copyfile(f"{file}/{filename}", f"{save_full_path}/{filename}")

    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    
    # 2. 提取关键点和 bbox
    print("extracting landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, args.bbox_shift)
    
    # 3. 生成 latents 和 mask
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    
    for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = min(y2 + args.extra_margin, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
            
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)

    mask_coords_list_cycle = []
    for i, frame in enumerate(tqdm(frame_list)):
        cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)
        x1, y1, x2, y2 = coord_list[i]
        mode = args.parsing_mode if args.version == "v15" else "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
        mask_coords_list_cycle.append(crop_box)

    # 4. 保存文件
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
    torch.save(input_latent_list, latents_out_path)
    print(f"Avatar '{avatar_id}' generated successfully at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MuseTalk Avatar Assets")
    parser.add_argument("--file", type=str, required=True, help="Path to video or image folder")
    parser.add_argument("--avatar_id", type=str, required=True, help="Name for the avatar")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", default='jaw')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading models on {device}...")
    vae, unet, pe = load_all_model(device=device)
    vae.vae = vae.vae.half().to(device)
    
    if args.version == "v15":
        fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    else:
        fp = FaceParsing()

    create_musetalk_human(args.file, args.avatar_id, args, vae, fp)