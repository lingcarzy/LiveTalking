
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def read_imgs(img_list):
    def load_image(index, img_path):
        return index, cv2.imread(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    max_workers = min(max(1, len(img_list)), max(1, min(8, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            if img is None:
                raise FileNotFoundError(f"failed to read image: {img_list[idx]}")
            frames[idx] = img
    return frames

def mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 