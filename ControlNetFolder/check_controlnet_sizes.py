from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

from PIL import Image
import torch
from matplotlib import pyplot as plt

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
).to('cuda')

orig_forward = pipe.controlnet.forward

def patched_forward(self, *args, **kwargs):
    out = orig_forward(*args, **kwargs)
    
    down_block_res_samples = out[0]
    mid_block_res_sample = out[1]

    print("ControlNet Down Blocks:")
    for i, sample in enumerate(down_block_res_samples):
        print(f"  Down block {i}: {sample.shape}")
    
    print("ControlNet Mid Block:", mid_block_res_sample.shape)

    return out

pipe.controlnet.forward = patched_forward.__get__(pipe.controlnet, type(pipe.controlnet))

image = load_image("./512.png")
mask = load_image("./512.png")
conditioning = load_image("./512.png")

text_prompt = ""

generator = torch.manual_seed(1)

new_image = pipe(
    text_prompt,
    num_inference_steps=20,
    generator=generator,
    image=image,
    control_image=conditioning,
    controlnet_conditioning_scale = 0.5,
    mask_image=mask
).images[0]

new_image.save("./fake_output.png")
