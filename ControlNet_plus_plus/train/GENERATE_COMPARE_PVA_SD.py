import sys
sys.path.append("diffusers_new/src")

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os

# ========== CONFIG ==========
DATASET_NAME_MASKS = "Milocas/celebahq_masked"
IMG_DIR = "./COMPARE_PVA/images"
MASK_DIR = "./COMPARE_PVA/masks"
SAVE_DIR = "./COMPARE_PVA/output_base"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset_masks = load_dataset(DATASET_NAME_MASKS, split="train")

# ========== LOAD MODELS ==========
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()


# ========== PROCESS ALL FILES ==========
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
print(f"Found {len(img_files)} images.")

for idx, img_file in enumerate(img_files):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMG_DIR, img_file)

    image = Image.open(img_path).convert("RGB").resize((512, 512))

    print(f"[{idx+1}/{len(img_files)}] Processing {base_name}")

    for m in range(4):
        mask_name = f"{base_name}_mask{m}.png"
        mask_path = os.path.join(MASK_DIR, mask_name)
        if not os.path.exists(mask_path):
            print(f"Missing {mask_name}, skipping...")
            continue

        mask = Image.open(mask_path).resize((512, 512))

        with torch.no_grad(), torch.autocast(device.type):
            result = pipe_base(
                prompt="",
                image=image,
                mask_image=mask,
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(2023),
            ).images[0]

        save_name = f"{base_name}_mask{m}_out.png"
        result.save(os.path.join(SAVE_DIR, save_name))