import sys
sys.path.append("diffusers_new/src")

import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

from diffusers import StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from backbones.iresnet import iresnet100

# ========== CONFIG ==========
CONTROLNET_PATH = "../../identity_controlnet_final"
DATASET_NAME = "Milocas/celebahq_clean"
IMG_DIR = "./DATASET_EVAL/images_ocluded_table"
MASK_DIR = "./DATASET_EVAL/masks_table"
NUM_SAMPLES = 10
SAVE_DIR = "./COMPARE_IDS_CONTROLNET"

MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"

os.makedirs(SAVE_DIR, exist_ok=True)
grid_dir = os.path.join(SAVE_DIR, f"grid_{NUM_SAMPLES}x{NUM_SAMPLES}")
os.makedirs(grid_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ARCFACE SETUP ==========
class ARCFACE(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(device))
            return F.normalize(x, p=2, dim=1)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def compute_embedding(img: Image.Image, model: ARCFACE):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return model(tensor).squeeze(0).cpu()

arcface = ARCFACE(MODEL_PATH)

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
samples = [dataset[i] for i in range(NUM_SAMPLES)]

occluded_imgs, masks, embeddings, names = [], [], [], []
for i, sample in enumerate(samples):
    name_img = f"{i}.png"
    names.append(name_img)

    image_masked_path = os.path.join(IMG_DIR, name_img)
    mask_path = os.path.join(MASK_DIR, name_img)

    occluded_imgs.append(Image.open(image_masked_path).convert("RGB").resize((512, 512)))
    masks.append(Image.open(mask_path).convert("L").resize((512, 512)))
    embeddings.append(torch.from_numpy(np.load("../../" + sample["condition"])).squeeze(0))

embeddings = torch.stack(embeddings)

# ========== LOAD CONTROLNET ==========
print("Loading ControlNet")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== GENERATE GRID ==========
similarities = np.zeros((NUM_SAMPLES, NUM_SAMPLES))

for i in range(NUM_SAMPLES):       # occluded images (rows)
    for j in range(NUM_SAMPLES):   # identity embeddings (columns)
        print(f"Generating [{i},{j}]...")

        with torch.no_grad(), torch.autocast(device.type):
            gen_img = pipe_controlnet(
                prompt="",
                image=occluded_imgs[i],
                mask_image=masks[i],
                control_image=embeddings[j].unsqueeze(0).to(device),
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(1000 + i*NUM_SAMPLES + j),
            ).images[0]

        gen_path = os.path.join(grid_dir, f"gen_row{i}_col{j}.png")
        gen_img.save(gen_path)

        emb_gen = compute_embedding(gen_img, arcface)

        sim = F.cosine_similarity(emb_gen, embeddings[j].cpu(), dim=0).item()
        similarities[i, j] = sim

# ========== SAVE COSINE SIMILARITIES ==========
df = pd.DataFrame(
    similarities,
    index=[f"occ_{i}" for i in range(NUM_SAMPLES)],
    columns=[f"id_{j}" for j in range(NUM_SAMPLES)]
)
df.to_csv(os.path.join(SAVE_DIR, f"sims_{NUM_SAMPLES}x{NUM_SAMPLES}.csv"))