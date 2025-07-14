import sys
sys.path.append("diffusers_new/src")

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
import os

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_masked"
CONTROLNET_PATH = "../../identity_controlnet_face_specific"
EMBEDDING_PREFIX = "../../"
NUM_IMAGES = 60
NUM_SEEDS = 100
SAVE_DIR = "./seed_randomness_outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== SETUP OUTPUT ==========
for model_type in ["base", "controlnet", "input"]:
    model_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)
    for i in range(NUM_IMAGES):
        os.makedirs(os.path.join(model_dir, f"{i:03d}"), exist_ok=True)

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")

# ========== LOAD MODELS ==========
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet...")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== PROCESS MULTIPLE IMAGES AND SEEDS ==========
print(f"Generating {NUM_SEEDS} variations for each of {NUM_IMAGES} images...")

for idx in range(NUM_IMAGES):
    print(f"\nProcessing image {idx+1}/{NUM_IMAGES}")
    sample = dataset[idx]
    image = sample["image"].resize((512, 512))
    mask = sample["mask"].resize((512, 512))
    embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

    # Save input image and mask once
    if not os.path.exists(os.path.join(SAVE_DIR, "input", f"{idx:03d}", "image.png")):
        image.save(os.path.join(SAVE_DIR, "input", f"{idx:03d}", "image.png"))
        mask.save(os.path.join(SAVE_DIR, "input", f"{idx:03d}", "mask.png"))

    for seed in range(NUM_SEEDS):
        seed_val = 1000 + 1000 * seed

        with torch.no_grad(), torch.autocast(device.type):
            result_base = pipe_base(
                prompt="",
                image=image,
                mask_image=mask,
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(seed_val),
            ).images[0]

            result_controlnet = pipe_controlnet(
                prompt="",
                image=image,
                mask_image=mask,
                control_image=embedding,
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(seed_val),
            ).images[0]

        result_base.save(os.path.join(SAVE_DIR, "base", f"{idx:03d}", f"seed_{seed_val}.png"))
        result_controlnet.save(os.path.join(SAVE_DIR, "controlnet", f"{idx:03d}", f"seed_{seed_val}.png"))

print(f"\nAll seed-variation results saved in: {SAVE_DIR}")
