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
CONTROLNET_PATH = "../../identity_controlnet"
EMBEDDING_PREFIX = "../../"
NUM_SAMPLES = 100
SAVE_DIR = "./comparison_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# ========== LOAD MODELS ==========
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

print("Loading custom ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_PATH, torch_dtype=torch.float32
).to(device)

print("Loading custom ControlNet pipeline...")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== PROCESS MULTIPLE SAMPLES ==========
for i in range(NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Processing sample...")
    sample = dataset[i]
    name = f"sample_{i:03d}"

    # Load and preprocess inputs
    image = sample["image"].resize((512, 512))
    mask = sample["mask"].resize((512, 512))
    embedding_path = sample["condition"]
    embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + embedding_path)).unsqueeze(0).to(device)
    
    # Run base inpainting
    with torch.no_grad(), torch.autocast(device.type):
        result_base = pipe_base(
            prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

    # Run ControlNet inpainting
    with torch.no_grad(), torch.autocast(device.type):
        result_controlnet = pipe_controlnet(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=embedding,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

    # Save individual outputs
    # image.save(os.path.join(SAVE_DIR, f"{name}_input_masked.png"))
    # result_base.save(os.path.join(SAVE_DIR, f"{name}_base.png"))
    # result_controlnet.save(os.path.join(SAVE_DIR, f"{name}_controlnet.png"))

    # Save side-by-side comparison
    comparison = Image.new("RGB", (512 * 3, 512))
    comparison.paste(image, (0, 0))
    comparison.paste(result_base, (512, 0))
    comparison.paste(result_controlnet, (512 * 2, 0))
    comparison.save(os.path.join(SAVE_DIR, f"{name}_comparison.png"))

print(f"\nSaved results to: {SAVE_DIR}")
