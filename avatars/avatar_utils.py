import glob
import os
import pickle

import torch

from utils.device import initialize_device
from utils.image import read_imgs, mirror_index
from utils.logger import logger


IMAGE_GLOB_PATTERN = '*.[jpJP][pnPN]*[gG]'


def get_inference_device():
    device = initialize_device()
    logger.info('Using {} for inference.'.format(device))
    return device


def get_avatar_path(avatar_id: str) -> str:
    return os.path.join('./data/avatars', avatar_id)


def _numeric_stem_sort_key(file_path: str) -> int:
    return int(os.path.splitext(os.path.basename(file_path))[0])


def load_sorted_images(image_dir: str):
    image_paths = glob.glob(os.path.join(image_dir, IMAGE_GLOB_PATTERN))
    image_paths = sorted(image_paths, key=_numeric_stem_sort_key)
    return read_imgs(image_paths)


def load_pickle_file(file_path: str):
    with open(file_path, 'rb') as file_obj:
        return pickle.load(file_obj)


def load_torch_file(file_path: str, map_location=None):
    if map_location is None:
        return torch.load(file_path)
    return torch.load(file_path, map_location=map_location)


def get_mirror_batch_indices(length: int, start_index: int, batch_size: int) -> list[int]:
    return [mirror_index(length, start_index + offset) for offset in range(batch_size)]


def warm_up_avatar_model(build_inputs, forward_fn, log_message: str = 'warmup model...'):
    logger.info(log_message)
    return forward_fn(*build_inputs())