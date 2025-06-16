from diffusers import DDIMScheduler
from diffusers_new.src.diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("Milocas/celebahq_masked", split="train")
sample = dataset[0]

image = sample["image"].resize((512, 512))
mask = sample["mask"].resize((512, 512))
embedding_path = sample["condition"]
embedding = torch.from_numpy(np.load("../../"+embedding_path)).unsqueeze(0).to(device) 

controlnet = ControlNetModel.from_pretrained(
    "../../identity_controlnet", 
    torch_dtype=torch.float32
).to(device)

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

result.save("inpainted_result.png")