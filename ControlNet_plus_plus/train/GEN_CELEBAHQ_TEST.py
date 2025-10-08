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
CONTROLNET_PATH = "../../identity_controlnet_final"
DATASET_NAME = "Milocas/celebahq_clean"
MASKED_IMAGE_PATH = "./DATASETS_FOR_TESTING/CELEBAHQ/eyes_masked_images"
MASK_DIR = "./mask.png"
EMBEDDING_PREFIX = "../../"
SAVE_DIR = "./TEST_CELEBAHQ/TEST"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.join(SAVE_DIR, "base")
controlnet_dir = os.path.join(SAVE_DIR, "controlnet")

for d in [base_dir, controlnet_dir]:
    os.makedirs(d, exist_ok=True)

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")

# ========== LOAD MODELS ==========
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, 
    safety_checker = None, requires_safety_checker = False

).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

print("Loading ControlNet")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, 
    safety_checker = None, requires_safety_checker = False

).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

NUM_SAMPLES = 2953

# ========== PROCESS MULTIPLE SAMPLES ==========
for i in range(NUM_SAMPLES):
    sample = dataset[i]

    embedding_path = EMBEDDING_PREFIX + sample["condition"]
    embedding_name = os.path.splitext(os.path.basename(embedding_path))[0]
    image_masked_path = os.path.join(MASKED_IMAGE_PATH, f"{embedding_name}.jpg")

    if not os.path.exists(image_masked_path):
        print(f"Masked image not found: {image_masked_path}, skipping...")
        continue
    
    print(f"[{i+1}/{len(dataset)}] Processing sample {embedding_name}...")

    image = Image.open(image_masked_path).convert("RGB").resize((512, 512))
    mask = Image.open(MASK_DIR).resize((512, 512))
    embedding = torch.from_numpy(np.load(embedding_path)).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.autocast(device.type):
        # Base pipeline
        result_base = pipe_base(
            prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(2000+i),
        ).images[0]

        # ControlNet pipeline
        result_controlnet = pipe_controlnet(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=embedding,
            num_inference_steps=25,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device).manual_seed(2000+i),
        ).images[0]

    # Save results
    name = f"{embedding_name}.jpg"
    result_base.save(os.path.join(base_dir, name))
    result_controlnet.save(os.path.join(controlnet_dir, name))

print(f"\nAll results saved to: {SAVE_DIR}")