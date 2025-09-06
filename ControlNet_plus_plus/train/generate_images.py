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
DATASET_NAME = "Milocas/celebahq_masked"
CONTROLNET_PATH = "../../identity_controlnet_final"
EMBEDDING_PREFIX = "../../"
NUM_SAMPLES = 2953
SAVE_DIR = "./comparison_outputs_random_seed_augmented_control"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.join(SAVE_DIR, "base")
controlnet_dir = os.path.join(SAVE_DIR, "controlnet")
input_dir = os.path.join(SAVE_DIR, "input")

for d in [base_dir, controlnet_dir, input_dir]:
    os.makedirs(d, exist_ok=True)
    
# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")

# ========== LOAD MODELS ==========
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

print("Loading ControlNet Masked Embeddings")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== PROCESS MULTIPLE SAMPLES ==========
processed = len(os.listdir(base_dir))
print(f"Resuming from sample {processed}")

for i in range(processed, NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Processing sample...")
    sample = dataset[i]
    name = f"{i:03d}.png"

    image = sample["image"].resize((512, 512))
    mask = sample["mask"].resize((512, 512))

    embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device.type):
        result_base = pipe_base(
            prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(1000+i),
        ).images[0]

        result_controlnet = pipe_controlnet(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=embedding,
            controlnet_conditioning_scale=0.7,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(1000+i),
        ).images[0]

    image.save(os.path.join(input_dir, name))
    result_base.save(os.path.join(base_dir, name))
    result_controlnet.save(os.path.join(controlnet_dir, name))

print(f"\nAll results saved to: {SAVE_DIR}")