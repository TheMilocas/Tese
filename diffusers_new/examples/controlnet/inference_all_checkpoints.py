import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import os
import glob

TRAINING_DIR = "./trained_controlnet_2"  

OUTPUT_DIR = os.path.join(TRAINING_DIR, "inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

control_image_path = "./conditioning_image_1.png" 
control_image = load_image(control_image_path)

prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)

checkpoints = sorted(glob.glob(os.path.join(TRAINING_DIR, "checkpoint-*")), key=lambda x: int(x.split('-')[-1]))

for checkpoint_path in checkpoints:
    checkpoint_name = os.path.basename(checkpoint_path)
    controlnet_path = os.path.join(checkpoint_path, "controlnet") 

    print(f"Running inference with {checkpoint_name}")

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)#.to("cuda")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
    ).to("cuda")
    
    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]

    output_path = os.path.join(OUTPUT_DIR, f"{checkpoint_name}.png")
    image.save(output_path)

    print(f"Saved inference result: {output_path}")