import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from backbones.iresnet import iresnet100
from datasets import load_dataset

# ===== CONFIG =====
CELEBA_CLEAN_DIR = "Milocas/celebahq_clean"   
CELEBA_MASKED_DIR = "Milocas/celebahq_masked"
FFHQ_DATASET = "Ryan-sjtu/ffhq512-caption"
FFHQ_MASKED_DIR = "DATASET_EVAL/FFHQ/ffhq_with_celebahq_masks"
OUTPUT_DIR = "DATASET_EVAL/COMBINED"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 2953
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_metrics.csv")

print("Loading datasets...")
celeba_masked = load_dataset(CELEBA_MASKED_DIR, split="train")
celeba_clean = load_dataset(CELEBA_CLEAN_DIR, split="train")
ffhq = load_dataset(FFHQ_DATASET, split="train")

# ===== ARCFACE SETUP =====
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ARCFACE/models/R100_MS1MV3/backbone.pth"))
arcface = ARCFACE(arcface_model_path)

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize([0.5], [0.5])
])

def pil_from_field(field):
    """Convert HF image/mask field to PIL.Image if possible."""
    if isinstance(field, Image.Image):
        return field
    # huggingface datasets often return dict with 'path' or numpy array
    if isinstance(field, dict):
        # try 'path' or 'bytes'
        if 'path' in field and os.path.exists(field['path']):
            return Image.open(field['path']).convert("RGB")
        if 'bytes' in field:
            from io import BytesIO
            return Image.open(BytesIO(field['bytes'])).convert("RGB")
    if isinstance(field, np.ndarray):
        return Image.fromarray(field)
    # last attempt: try to cast
    try:
        return Image.fromarray(np.array(field))
    except Exception:
        return None

# ===== UTILITIES =====
def cosine_similarity(a, b):
    sim = F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()
    return max(min(sim, 1.0), -1.0)  # clip to [-1,1]

def mask_area_px(mask_img: Image.Image):
    # assumes black pixels represent mask
    mask_arr = np.array(mask_img.convert("L"))
    return int(np.sum(mask_arr < 10))

def load_images_from_dir(directory):
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith((".png",".jpg",".jpeg"))])
    paths = [os.path.join(directory, f) for f in files]
    return paths

# ===== LOAD DATA =====
ffhq_masked_files = load_images_from_dir(FFHQ_MASKED_DIR)

results = []

# ===== CELEBA CLEAN vs MASKED =====

limit = min(len(celeba_clean), len(celeba_masked))
if NUM_SAMPLES is not None:
    limit = min(limit, NUM_SAMPLES)
    
ffhq_masked_files = load_images_from_dir(FFHQ_MASKED_DIR)

results = []

# ===== CELEBA CLEAN vs MASKED =====
for idx in tqdm(range(limit), desc="Computing embeddings CelebA clean vs masked"):
    try:
        # load clean and masked images
        clean_img = pil_from_field(celeba_clean[idx]["image"])
        masked_img = pil_from_field(celeba_masked[idx]["image"])
        mask = pil_from_field(celeba_masked[idx]["mask"])
        
        if clean_img is None or masked_img is None:
            print(f"Could not load images for idx {idx}, skipping")
            continue

        clean_img = clean_img.resize((512, 512))
        masked_img = masked_img.resize((512, 512))
        mask = mask.resize((512, 512))

        # compute embeddings
        clean_t = transform(clean_img).unsqueeze(0).to(DEVICE)
        masked_t = transform(masked_img).unsqueeze(0).to(DEVICE)

        emb_clean = arcface(clean_t).squeeze(0)
        emb_masked = arcface(masked_t).squeeze(0)

        # similarity and derived metrics
        sim = cosine_similarity(emb_clean, emb_masked)
        identity_hidden = 1.0 - sim

        # compute mask area and efficiency if needed (assumes black pixels are occluded)
        mask = pil_from_field(celeba_masked[idx]["mask"]).resize((512, 512))
        mask_area_px = int(np.sum(np.array(mask.convert("L")) > 128))  # white pixels represent occlusion
        h, w = 512, 512
        relative_area = mask_area_px / (h * w) if (h*w) > 0 else 0
        efficiency = identity_hidden / relative_area if relative_area > 0 else 0

        # store results
        results.append({
            "dataset": "celebahq",
            "image_id": idx,
            "similarity": sim,
            "identity_hidden": identity_hidden,
            "relative_area": relative_area,
            "efficiency": efficiency,
            "mask_area_px": mask_area_px
        })

    except Exception as e:
        print(f"Failed on idx {idx}: {e}")
        continue


# ===== FFHQ MASKED (already applied) =====
for idx, ff_path in enumerate(tqdm(ffhq_masked_files, desc="FFHQ masked images")):
    try:
        clean_img = pil_from_field(ffhq[idx]["image"])
        clean_img = clean_img.resize((512, 512))
        clean_t = transform(clean_img).unsqueeze(0).to(DEVICE)
        emb_clean = arcface(clean_t).squeeze(0)
        
        ff_img = Image.open(ff_path).convert("RGB").resize((512,512))
        ff_t = transform(ff_img).unsqueeze(0).to(DEVICE)
        emb_ff = arcface(ff_t).squeeze(0)

        # similarity and derived metrics
        sim = cosine_similarity(emb_clean, emb_ff)
        identity_hidden = 1.0 - sim

        # compute mask area and efficiency if needed (assumes black pixels are occluded)
        mask_area_px = int(np.sum(np.array(masked_img.convert("L")) < 10))  # black pixels
        h, w = 512, 512
        relative_area = mask_area_px / (h * w) if (h*w)>0 else 0
        efficiency = identity_hidden / relative_area if relative_area > 0 else 0

        results.append({
            "dataset": "ffhq",
            "image_id": idx,
            "similarity": sim,
            "identity_hidden": identity_hidden,
            "relative_area": relative_area,
            "efficiency": efficiency
        })
    except Exception as e:
        print(f"FFHQ idx {idx} failed: {e}")

# ===== SAVE CSV =====
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"Saved combined CSV: {CSV_PATH}")