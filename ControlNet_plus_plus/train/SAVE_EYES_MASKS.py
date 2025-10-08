import sys
import os
notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, "../../")))
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
from backbones.iresnet import iresnet100
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
OUTPUT_DIR = "DATASET_EVAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASK_NAMES = ["eyes", "mouth", "nose"]

FACEMESH_INDICES = {
    "eyes": [33, 133, 160, 159, 158, 157, 173, 246,   # right eye contour
             362, 263, 387, 386, 385, 384, 398, 466], # left eye contour
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405,
              321, 375, 291, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95, 185, 40, 39,
              37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78],   # full lips
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5,
             4, 45, 220, 275, 440, 344, 278, 331]   # nose bridge + tip
}


# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# ========== ARCFACE SETUP ==========
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(ARCFACE, self).__init__()
        self.device = torch.device('cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

# ========== UTILITIES ==========
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize([0.5], [0.5])
])

def create_rect_mask(image, landmarks, indices, padding=10, scale=1.0):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Box center
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    bw, bh = (x_max - x_min), (y_max - y_min)

    # Expand box
    bw = int(bw * scale)
    bh = int(bh * scale)

    x_min = max(cx - bw // 2 - padding, 0)
    y_min = max(cy - bh // 2 - padding, 0)
    x_max = min(cx + bw // 2 + padding, w)
    y_max = min(cy + bh // 2 + padding, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    area = (x_max - x_min) * (y_max - y_min)
    return mask, area


def apply_mask_with_landmarks(image_np):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        masked_outputs = {}

        for region, indices in FACEMESH_INDICES.items():
            if region == "eyes":
                mask, area = create_rect_mask(image_np, landmarks, indices, padding=20, scale=1.5)
            else:
                mask, area = create_rect_mask(image_np, landmarks, indices, padding=10, scale=1.0)
            mask_3ch = cv2.merge([mask, mask, mask])
            masked = cv2.bitwise_and(image_np, cv2.bitwise_not(mask_3ch))
            masked_outputs[region] = {"image": masked, "area": area}

        return masked_outputs


#========== MAIN ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ARCFACE/models/R100_MS1MV3/backbone.pth"))
arcface = ARCFACE(arcface_model_path)

results_list = []
NUM_SAMPLES = 2953
for idx in tqdm(NUM_SAMPLES, desc="Processing"):
    try:
        sample = dataset[idx]
        image = sample["image"].resize((512, 512))

        original_tensor = transform(image).unsqueeze(0).to(device)
        original_embedding = arcface(original_tensor).squeeze(0)

        image_np = np.array(image)  # RGB
        masked_versions = apply_mask_with_landmarks(image_np)
        if masked_versions is None:
            print(f"No face detected in sample {idx}")
            continue

        row = {"image_id": idx}
        h, w, _ = image_np.shape

        for region, masked_data in masked_versions.items():
            masked_img = masked_data["image"]
            mask_area = masked_data["area"]

            masked_pil = Image.fromarray(masked_img)
            region_dir = os.path.join(OUTPUT_DIR, "masks", region)
            os.makedirs(region_dir, exist_ok=True)
            masked_path = os.path.join(region_dir, f"sample_{idx}.png")
            masked_pil.save(masked_path)

            masked_tensor = transform(masked_pil).unsqueeze(0).to(device)
            emb = arcface(masked_tensor).squeeze(0)
            sim = cosine_similarity(original_embedding, emb)

            # === Metrics ===
            identity_hidden = 1 - sim
            relative_area = mask_area / (h * w)
            efficiency = identity_hidden / relative_area if relative_area > 0 else 0

            # Store results
            row[region] = sim
            row[f"{region}_mask_area"] = mask_area
            row[f"{region}_identity_hidden"] = identity_hidden
            row[f"{region}_efficiency"] = efficiency

        results_list.append(row)

    except Exception as e:
        print(f"Failed on sample {idx}: {e}")
        continue

# Save results
df = pd.DataFrame(results_list)
csv_path = os.path.join(OUTPUT_DIR, "similarities.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved similarities to {csv_path}")

# ---------- Evaluation Plots ----------
plt.figure(figsize=(10,6))
for region in MASK_NAMES:
    if region in df.columns:
        plt.hist(df[region], bins=30, alpha=0.5, label=region)
plt.legend()
plt.title("Cosine Similarity Distributions by Masked Region")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "similarity_distributions.png"))

# Boxplot similarities
plt.figure(figsize=(8,6))
df.boxplot(column=MASK_NAMES)
plt.title("Boxplot of Similarities per Region")
plt.savefig(os.path.join(OUTPUT_DIR, "similarity_boxplots.png"))

# Boxplot efficiencies
plt.figure(figsize=(8,6))
df.boxplot(column=[f"{r}_efficiency" for r in MASK_NAMES])
plt.title("Mask Efficiency per Region (Identity Hidden / Area)")
plt.ylabel("Efficiency")
plt.savefig(os.path.join(OUTPUT_DIR, "mask_efficiency_boxplot.png"))

# Scatter Area vs Identity Hidden
plt.figure(figsize=(8,6))
for region in MASK_NAMES:
    plt.scatter(df[f"{region}_mask_area"], df[f"{region}_identity_hidden"],
                alpha=0.5, label=region)
plt.xlabel("Mask Area (px)")
plt.ylabel("Identity Hidden (1 - cosine similarity)")
plt.title("Tradeoff: Area vs Identity Hidden")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "mask_tradeoff.png"))

print(f"Plots saved in {OUTPUT_DIR}")
