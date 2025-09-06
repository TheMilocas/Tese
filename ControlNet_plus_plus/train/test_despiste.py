import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from datasets import load_dataset

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel

from backbones.iresnet import iresnet100
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

# ========== CONFIG ==========
CONTROLNET_PATH = "../../identity_controlnet_final"
EMB_PATH = "./FFHQ_inputs/input_embeddings/000.npy"
IMG_PATH = "./FFHQ_inputs/input/000.png"
MASK_PATH = "./mask.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ArcFace Model ==========
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(ARCFACE, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface_model_path = os.path.abspath("../../ARCFACE/models/R100_MS1MV3/backbone.pth")
arcface = ARCFACE(arcface_model_path)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def compute_embedding(img):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return arcface(tensor).squeeze(0).cpu()

def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

# ========== Load Pipelines ==========
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

image = Image.open(IMG_PATH).convert("RGB")
mask = Image.open(MASK_PATH).resize((512, 512))
emb = torch.from_numpy(np.load(EMB_PATH)).unsqueeze(0).to(device)

seed = 1000
generator = torch.Generator(device).manual_seed(seed)

with torch.no_grad(), torch.autocast(device.type):

    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_1 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        generator=generator
    ).images[0]
    
    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_2 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        generator=generator
    ).images[0]

emb_1 = compute_embedding(controlnet_img_1)
emb_2 = compute_embedding(controlnet_img_2)
comp_equal_1 = cosine_similarity(emb_1, emb_2)

with torch.no_grad(), torch.autocast(device.type):

    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_CS_05_1 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        controlnet_conditioning_scale=0.5,
        generator=generator
    ).images[0]

    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_CS_05_2 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        controlnet_conditioning_scale=0.5,
        generator=generator
    ).images[0]

emb_3 = compute_embedding(controlnet_img_CS_05_1)
emb_4 = compute_embedding(controlnet_img_CS_05_2)
comp_equal_2 = cosine_similarity(emb_3, emb_4)

with torch.no_grad(), torch.autocast(device.type):

    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_CS_1_1 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        controlnet_conditioning_scale=1.0,
        generator=generator
    ).images[0]

    generator = torch.Generator(device).manual_seed(seed)
    
    controlnet_img_CS_1_2 = pipe_controlnet(
        prompt="",
        image=image,
        mask_image=mask,
        control_image=emb,
        num_inference_steps=25,
        controlnet_conditioning_scale=1.0,
        generator=generator
    ).images[0]

emb_5 = compute_embedding(controlnet_img_CS_1_1)
emb_6 = compute_embedding(controlnet_img_CS_1_2)
comp_equal_3 = cosine_similarity(emb_5, emb_6) 

# comp_equal must be 1 in every case

comp_def_05 = cosine_similarity(emb_1, emb_3) 

comp_def_1 = cosine_similarity(emb_1, emb_5) 

sims = [comp_equal_1, comp_equal_2, comp_equal_3]

for i, sim in enumerate(sims, 1):
    print(f"Pair {i} similarity: {sim:.4f}")

print(f"Comparison between default values and 0.5:{comp_def_05:.4f} and default and 1:{comp_def_1:.4f}")
