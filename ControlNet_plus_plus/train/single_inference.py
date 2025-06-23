import sys
sys.path.append("diffusers_new/src")

from diffusers import StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os
import random

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_masked"
CONTROLNET_PATH = "../../identity_controlnet_no_masks"
EMBEDDING_PREFIX = "../../"
SAVE_DIR = "./sample_output"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset(DATASET_NAME, split="train")
sample = random.choice(dataset)

image = sample["image"].resize((512, 512))
mask = sample["mask"].resize((512, 512))
embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

with torch.no_grad(), torch.autocast(device.type):
    result = pipe(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=embedding,
        num_inference_steps=25,
        generator=torch.Generator(device).manual_seed(42),
    ).images[0]

def create_output_image(images, labels, save_path):
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

create_output_image(
    images=[image, mask.convert("RGB"), result],
    labels=["Masked Input", "Mask", "Inpainted"],
    save_path=os.path.join(SAVE_DIR, "comparison.png")
)

print(f"\n Image saved to: {os.path.join(SAVE_DIR, 'comparison.png')}")
