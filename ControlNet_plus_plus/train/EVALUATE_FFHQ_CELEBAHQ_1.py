import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from backbones.iresnet import iresnet100
from datasets import load_dataset

# ===== CONFIG =====
CELEBA_CSV = "DATASET_EVAL/COMBINED/combined_metrics.csv" 
FFHQ_DATASET = "Ryan-sjtu/ffhq512-caption"
FFHQ_MASKED_DIR = "../../datasets/ffhq/masked_images" 
OUTPUT_DIR = "DATASET_EVAL/COMBINED"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_metrics_with_ffhq.csv")

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
    try:
        return Image.fromarray(np.array(field))
    except Exception:
        return None

def cosine_similarity(a, b):
    sim = F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()
    return max(min(sim, 1.0), -1.0)

def load_images_from_dir(directory):
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith((".png",".jpg",".jpeg"))])
    paths = [os.path.join(directory, f) for f in files]
    return paths

# ===== LOAD DATA =====
print("Loading data...")
celeba_df = pd.read_csv(CELEBA_CSV)
ffhq = load_dataset(FFHQ_DATASET, split="train")
ffhq_masked_files = load_images_from_dir(FFHQ_MASKED_DIR)

# Ensure same length
limit = 2953

ffhq_results = []

# ===== FFHQ CLEAN vs MASKED =====
for idx in tqdm(range(limit), desc="Computing embeddings FFHQ clean vs masked"):
    try:
        clean_img = pil_from_field(ffhq[idx]["image"]).resize((512, 512))
        clean_t = transform(clean_img).unsqueeze(0).to(DEVICE)
        emb_clean = arcface(clean_t).squeeze(0)

        ff_img = Image.open(ffhq_masked_files[idx]).convert("RGB").resize((512,512))
        ff_t = transform(ff_img).unsqueeze(0).to(DEVICE)
        emb_ff = arcface(ff_t).squeeze(0)

        sim = cosine_similarity(emb_clean, emb_ff)
        identity_hidden = 1.0 - sim

        # compute mask area (only once, from CelebA masks) → take it from CelebA DF
        mask_area_px = celeba_df.loc[idx, "mask_area_px"]
        relative_area = celeba_df.loc[idx, "relative_area"]

        efficiency = identity_hidden / relative_area if relative_area > 0 else 0

        ffhq_results.append({
            "dataset_ffhq": "ffhq",
            "image_id_ffhq": idx,
            "similarity_ffhq": sim,
            "identity_hidden_ffhq": identity_hidden,
            "relative_area_ffhq": relative_area,
            "efficiency_ffhq": efficiency
        })

    except Exception as e:
        print(f"FFHQ idx {idx} failed: {e}")

# ===== MERGE CELEBA + FFHQ SIDE BY SIDE =====
ffhq_df = pd.DataFrame(ffhq_results)

# Reset indexes to align row by row
celeba_df = celeba_df.reset_index(drop=True)
ffhq_df = ffhq_df.reset_index(drop=True)

combined_df = pd.concat([celeba_df, ffhq_df], axis=1)

# Save
combined_df.to_csv(CSV_PATH, index=False)
print(f"Saved combined CSV with CelebA and FFHQ side-by-side: {CSV_PATH}")