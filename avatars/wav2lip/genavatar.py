from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from avatars.wav2lip import face_detection
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--nosmooth', default=False, action='store_true')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
parser.add_argument('--face_det_batch_size', type=int, default=32, help='Batch size for face detection')
parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for I/O')
parser.add_argument('--resize_factor', type=int, default=1, help='Resize factor for faster detection')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

# ==================== 优化1: 单次读取视频帧 ====================
def extract_video_frames(vid_path, cut_frame=10000000):
    """读取视频帧到内存，供后续检测和落盘共用，避免二次磁盘读取。"""
    cap = cv2.VideoCapture(vid_path)
    frames = []
    count = 0

    print("Reading video frames...")
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "LiveTalking", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
        frames.append(frame)
        count += 1
    cap.release()
    return frames


def save_frames_parallel(frames, output_dir, num_workers=4):
    """并行保存帧，CPU+磁盘 I/O 可与 GPU 检测重叠。"""
    def save_frame(args):
        idx, frame = args
        cv2.imwrite(f"{output_dir}/{idx:08d}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    print(f"Saving {len(frames)} frames with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(save_frame, enumerate(frames)), total=len(frames)))

# ==================== 优化3: GPU批量人脸检测优化 ====================
class FaceDetector:
    """单例检测器，避免重复创建"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                flip_input=False, 
                device=device
            )
        return cls._instance
    
    def get_detections(self, images, batch_size):
        predictions = []
        for i in range(0, len(images), batch_size):
            batch = np.array(images[i:i + batch_size])
            # 如果图片太大，先缩小检测
            if args.resize_factor > 1:
                h, w = batch.shape[1:3]
                batch = np.array([cv2.resize(img, (w//args.resize_factor, h//args.resize_factor)) 
                                for img in batch])
            predictions.extend(self.detector.get_detections_for_batch(batch))
        return predictions

def get_smoothened_boxes(boxes, T):
    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    for i in range(len(boxes)):
        window = boxes[i:i + T] if i + T <= len(boxes) else boxes[len(boxes) - T:]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect_optimized(images, num_workers=4):
    """优化的人脸检测"""
    detector = FaceDetector()
    batch_size = args.face_det_batch_size
    
    # 自动调整batch size
    while batch_size >= 1:
        try:
            print(f"Trying batch size: {batch_size}")
            predictions = detector.get_detections(images, batch_size)
            break
        except RuntimeError as e:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU')
            batch_size //= 2
            torch.cuda.empty_cache()
            print(f'OOM, reducing batch size to: {batch_size}')
    
    # 并行处理裁剪
    pady1, pady2, padx1, padx2 = args.pads
    
    def process_detection(args):
        rect, image = args
        if rect is None:
            return None
        
        # 如果用了resize_factor，需要映射回原图坐标
        if args.resize_factor > 1:
            rect = [r * args.resize_factor for r in rect]
            
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        face_img = image[y1:y2, x1:x2]
        return [face_img, (y1, y2, x1, x2)]
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_detection, zip(predictions, images)))
    
    # 检查失败
    for i, res in enumerate(results):
        if res is None:
            cv2.imwrite('temp/faulty_frame.jpg', images[i])
            raise ValueError(f'Face not detected in frame {i}!')
    
    # 平滑处理
    boxes = np.array([r[1] for r in results])
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
        # 更新结果
        for i, (face_img, _) in enumerate(results):
            y1, y2, x1, x2 = boxes[i]
            results[i] = [images[i][int(y1):int(y2), int(x1):int(x2)], (y1, y2, x1, x2)]
    
    return results

# ==================== 优化3: 并行保存人脸图片 ====================
def save_faces_parallel(face_results, face_imgs_path, img_size, num_workers=4):
    """并行保存裁剪后的人脸"""
    def save_single(args):
        idx, (frame, coords) = args
        resized = cv2.resize(frame, (img_size, img_size))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized,
                   [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return coords
    
    print("Saving face images in parallel...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        coord_list = list(executor.map(
            save_single, 
            enumerate(face_results)
        ))
    return coord_list

# ==================== 主流程 ====================
if __name__ == "__main__":
    avatar_path = f"./data/avatars/{args.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    osmakedirs([avatar_path, full_imgs_path, face_imgs_path])
    print(args)
    
    # 1. 视频只读取一次，后续检测与 full_imgs 落盘共用同一批内存帧
    frames = extract_video_frames(args.video_path)

    # 2. full_imgs 落盘与 GPU 人脸检测并行进行，减少总墙钟时间
    with ThreadPoolExecutor(max_workers=1) as io_executor:
        full_frames_future = io_executor.submit(save_frames_parallel, frames, full_imgs_path, args.num_workers)

        # 3. 人脸检测（优化batch + GPU）
        face_det_results = face_detect_optimized(frames, num_workers=args.num_workers)
        full_frames_future.result()

    # 4. 保存人脸和坐标（并行）
    coord_list = save_faces_parallel(
        face_det_results, face_imgs_path, args.img_size, args.num_workers
    )

    # 5. 保存坐标
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
    
    print(f"Done! Processed {len(frames)} frames.")