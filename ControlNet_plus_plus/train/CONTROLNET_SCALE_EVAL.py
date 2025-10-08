import sys
sys.path.append("diffusers_new/src")

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

from diffusers import StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from backbones.iresnet import iresnet100
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
CONTROLNET_PATH = "../../identity_controlnet_final"
ARCFACE_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
MASKED_IMAGE_PATH = "../../datasets/celebahq/eyes_masked_images"
EMBEDDING_PREFIX = "../../"
MASK_DIR = "./mask.png"
NUM_SAMPLES = 100
SAVE_DIR = "./EVALUATE_CONTROLNET_SCALE"
IMG_SAVE_DIR = os.path.join(SAVE_DIR, "images")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

CONTROLNET_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 999, 2025, 7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== TRANSFORMS ==========
transform_arcface = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_fid = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def preprocess_fid(img):
    tensor = transform_fid(img)             # float in [0,1]
    tensor = (tensor * 255).to(torch.uint8) # uint8 in [0,255]
    return tensor.unsqueeze(0).to(DEVICE)

# ========== ARCFACE MODEL ==========
class ARCFACE(torch.nn.Module):
    def __init__(self, model_path, device=DEVICE):
        super().__init__()
        self.device = device
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface = ARCFACE(ARCFACE_PATH, device=DEVICE)
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
mfid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)

# ========== UTILITY FUNCTIONS ==========
def compute_embedding(img: Image.Image):
    img_tensor = transform_arcface(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    return arcface(img_tensor).squeeze(0)

def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

def compute_lpips(imgA, imgB):
    t1 = transforms.ToTensor()(imgA).unsqueeze(0).to(DEVICE) * 2 - 1
    t2 = transforms.ToTensor()(imgB).unsqueeze(0).to(DEVICE) * 2 - 1
    with torch.no_grad():
        return float(lpips_model(t1, t2))

# ========== LOAD MASK FOR mFID ==========
mask = Image.open(MASK_DIR).convert("L").resize((256,256))
mask_tensor = (transforms.ToTensor()(mask) > 0.5).to(DEVICE)  # boolean mask [1,H,W]

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# ========== LOAD CONTROLNET PIPELINE ==========
print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(DEVICE)
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(DEVICE)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== PROCESS SAMPLES ==========
results = []

for i in range(NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Processing sample...")
    
    sample = dataset[i]
    name = f"{i}.jpg"
    image_masked_path = os.path.join(MASKED_IMAGE_PATH, name)

    if not os.path.exists(image_masked_path):
        continue

    original_image = sample["image"].convert("RGB").resize((512, 512))
    image = Image.open(image_masked_path).convert("RGB").resize((512, 512))

    embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(DEVICE)

    for scale in CONTROLNET_SCALES:
        for seed in SEEDS:
            generator = torch.Generator(DEVICE).manual_seed(seed)
            with torch.no_grad(), torch.autocast(DEVICE.type):
                gen_image = pipe_controlnet(
                    prompt="", 
                    image=image,
                    mask_image=mask,
                    control_image=embedding,
                    num_inference_steps=25,
                    controlnet_conditioning_scale=scale,
                    generator=generator
                ).images[0]

            # Save generated image
            save_folder = os.path.join(IMG_SAVE_DIR, f"{name}/scale_{scale:.2f}")
            os.makedirs(save_folder, exist_ok=True)
            gen_image.save(os.path.join(save_folder, f"seed_{seed}.png"))

            # Compute embeddings and LPIPS
            emb_gen = compute_embedding(gen_image.convert("RGB"))
            lpips_val = compute_lpips(original_image, gen_image.convert("RGB"))
            cos_sim = cosine_similarity(embedding, emb_gen)

            # Compute FID and masked FID
            gen_tensor = preprocess_fid(gen_image.convert("RGB"))
            orig_tensor = preprocess_fid(original_image)

            # Global FID
            fid_metric.reset()
            fid_metric.update(gen_tensor, real=False)
            fid_metric.update(orig_tensor, real=True)
            fid_score = fid_metric.compute().item()

            # Masked FID
            masked_gen_tensor = gen_tensor * mask_tensor
            masked_orig_tensor = orig_tensor * mask_tensor
            mfid_metric.reset()
            mfid_metric.update(masked_gen_tensor, real=False)
            mfid_metric.update(masked_orig_tensor, real=True)
            mfid_score = mfid_metric.compute().item()

            results.append({
                "image": name,
                "scale": scale,
                "seed": seed,
                "cosine_similarity": cos_sim,
                "masked_lpips": lpips_val,
                "FID": fid_score,
                "mFID": mfid_score
            })

# Save CSV
df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "controlnet_scale_results_fid.csv")
df.to_csv(csv_path, index=False)
print(f"Saved metrics CSV to {csv_path}")