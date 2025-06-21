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
CONTROLNET_A_PATH = "../../identity_controlnet"
CONTROLNET_B_PATH = "../../identity_controlnet_no_masks"
EMBEDDING_PREFIX = "../../"
NUM_SAMPLES = 2953
SAVE_DIR = "./comparison_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.join(SAVE_DIR, "base")
masked_dir = os.path.join(SAVE_DIR, "masked")
unmasked_dir = os.path.join(SAVE_DIR, "unmasked")
input_dir = os.path.join(SAVE_DIR, "input")

for d in [base_dir, masked_dir, unmasked_dir, input_dir]:
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
controlnet_a = ControlNetModel.from_pretrained(CONTROLNET_A_PATH, torch_dtype=torch.float32).to(device)

print("Loading ControlNet Unmasked Embeddings")
controlnet_b = ControlNetModel.from_pretrained(CONTROLNET_B_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet Masked Embeddings")
pipe_controlnet_a = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet_a,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_controlnet_a.scheduler = DDIMScheduler.from_config(pipe_controlnet_a.scheduler.config)
pipe_controlnet_a.enable_model_cpu_offload()

print("Loading pipeline with ControlNet Unmasked Embeddings")
pipe_controlnet_b = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet_b,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_controlnet_b.scheduler = DDIMScheduler.from_config(pipe_controlnet_b.scheduler.config)
pipe_controlnet_b.enable_model_cpu_offload()

# ========== IMAGE COMPARISON UTILITY ==========
def create_comparison_image(images, labels, save_path):
    assert len(images) == len(labels), "Images and labels must match"

    width, height = images[0].size
    label_height = 30
    font_size = 20

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    result = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(result)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * width
        result.paste(img, (x, 0))
        text_width = draw.textlength(label, font=font)
        draw.text((x + (width - text_width) // 2, height + 5), label, fill="black", font=font)

    result.save(save_path)

# ========== PROCESS MULTIPLE SAMPLES ==========
processed = len(os.listdir(base_dir))
print(f"Resuming from sample {processed}")

for i in range(processed, NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Processing sample...")
    sample = dataset[i]
    name = f"sample_{i:03d}.png"

    # Load and resize
    image = sample["image"].resize((512, 512))
    mask = sample["mask"].resize((512, 512))

    embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device.type):
        result_base = pipe_base(
            prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

        result_masked = pipe_controlnet_a(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=embedding,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

        result_unmasked = pipe_controlnet_b(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=embedding,
            num_inference_steps=25,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

    # Create and save labeled comparison image
    # comparison_path = os.path.join(SAVE_DIR, f"{name}_comparison.png")
    # create_comparison_image(
    #     images=[image, result_base, result_masked, result_unmasked],
    #     labels=["Masked Input", "No ControlNet", "ControlNet Masked Embeddings", "ControlNet Unmasked Embeddings"],
    #     save_path=comparison_path
    # )
    image.save(os.path.join(input_dir, name))            # original masked input
    result_base.save(os.path.join(base_dir, name))       # no controlnet
    result_masked.save(os.path.join(masked_dir, name))   # controlnet A
    result_unmasked.save(os.path.join(unmasked_dir, name)) # controlnet B

print(f"\nAll results saved to: {SAVE_DIR}")