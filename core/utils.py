# core/utils.py
import torch
import os
import cv2
import pickle
import glob
from logger import logger
from tqdm import tqdm

# ============================================================================
# MuseTalk Utils
# ============================================================================
def load_musetalk_model():
    from musetalk.utils.utils import load_all_model
    logger.info("Loading MuseTalk model...")
    vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"))
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    
    from musetalk.whisper.audio2feature import Audio2Feature
    audio_processor = Audio2Feature(model_path="./models/whisper")
    
    return (vae, unet, pe, timesteps, audio_processor)

def load_musetalk_avatar(avatar_id):
    from core.render_loop import read_imgs # 复用通用读取函数
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path = f"{avatar_path}/latents.pt"
    mask_out_path = f"{avatar_path}/mask"
    mask_coords_path = f"{avatar_path}/mask_coords.pkl"

    input_latent_list_cycle = torch.load(latents_out_path)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
        
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')), 
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
        
    input_mask_list = sorted(glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    
    return frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle

def warm_up_musetalk(batch_size, model):
    logger.info('Warming up MuseTalk...')
    vae, unet, pe, timesteps, _ = model
    whisper_batch = torch.ones((batch_size, 50, 384), dtype=torch.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    
    with torch.no_grad():
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        vae.decode_latents(pred_latents)

# ============================================================================
# Wav2Lip Utils
# ============================================================================
def load_wav2lip_model(checkpoint_path):
    from wav2lip.models import Wav2Lip
    logger.info(f"Loading Wav2Lip model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def load_wav2lip_avatar(avatar_id):
    from core.render_loop import read_imgs
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
        
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    
    input_face_list = sorted(glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    
    return frame_list_cycle, face_list_cycle, coord_list_cycle

def warm_up_wav2lip(batch_size, model, model_res):
    logger.info('Warming up Wav2Lip...')
    device = next(model.parameters()).device
    img_batch = torch.ones(batch_size, 6, model_res, model_res).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    with torch.no_grad():
        model(mel_batch, img_batch)

# ============================================================================
# Ultralight Utils
# ============================================================================
def load_ultralight_model():
    from ultralight.audio2feature import Audio2Feature
    logger.info("Loading Ultralight audio processor...")
    return Audio2Feature()

def load_ultralight_avatar(avatar_id):
    from core.render_loop import read_imgs
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    
    # Load Model
    model_path = f"{avatar_path}/ultralight.pth"
    from ultralight.unet import Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(6, 'hubert').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
        
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    
    input_face_list = sorted(glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    
    return model.eval(), frame_list_cycle, face_list_cycle, coord_list_cycle

def warm_up_ultralight(batch_size, avatar, model_res):
    logger.info('Warming up Ultralight...')
    model, _, _, _ = avatar
    device = next(model.parameters()).device
    img_batch = torch.ones(batch_size, 6, model_res, model_res).to(device)
    mel_batch = torch.ones(batch_size, 16, 32, 32).to(device)
    with torch.no_grad():
        model(img_batch, mel_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames
# 辅助函数：音频播放 (用于 virtualcam 模式)
def play_audio_thread(quit_event, audio_queue):
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=8,
        output=True,
        output_device_index=1,
    )
    stream.start_stream()
    # while queue.qsize() <= 0:
    #     time.sleep(0.1)
    while not quit_event.is_set():
        stream.write(queue.get(block=True))
    stream.close()