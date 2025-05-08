from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image
import torch
import numpy as np

#trained_controlnet_inicial/checkpoint-500/controlnet
#lllyasviel/sd-controlnet-canny

controlnet = ControlNetModel.from_pretrained("zero_controlnet", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')

text_prompt=""

image = load_image("face_w_mask.png")
canny_image = load_image("canny_image.png")
mask_image = load_image("face_mask_mask.png")

generator = torch.manual_seed(1)
new_image = pipe(
    text_prompt,
    num_inference_steps=50,
    generator=generator,
    image=image,
    control_image=canny_image,
    mask_image=mask_image
).images[0]

new_image.save("./output1.png")