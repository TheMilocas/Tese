import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_single_mask"
SAVE_DIR = "./EYES_CELEBHQ_TRAIN_DEFAULT_EVAL/"

embedding_dir_base = os.path.join(SAVE_DIR, "embeddings_base")
embedding_dir_controlnet = os.path.join(SAVE_DIR, "embeddings_controlnet")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_SAMPLES = len(sorted([f for f in os.listdir(SAVE_DIR + "embeddings_base") if f.lower().endswith((".npy"))]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# ========== Cosine Similarity ==========
def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

# ========== INIT LISTS ==========
base_vs_original = []
controlnet_vs_original = []
valid_indices = []

# List all available embedding files
base_files = sorted([f for f in os.listdir(embedding_dir_base) if f.endswith(".npy")])

for fname in tqdm(base_files, desc="Processing"):
    try:
        base_path = os.path.join(embedding_dir_base, fname)
        controlnet_path = os.path.join(embedding_dir_controlnet, fname)

        if not os.path.exists(controlnet_path):
            print(f"Missing controlnet embedding for {fname}, skipping.")
            continue

        # Get numeric ID from filename
        sample_id = int(os.path.splitext(fname)[0])
        sample = dataset[sample_id]

        # Load embeddings (ensure all are on the same device)
        emb_original = torch.from_numpy(np.load("../../" + sample["condition"])).unsqueeze(0).to(device)
        emb_base = torch.from_numpy(np.load(base_path)).to(device)
        emb_controlnet = torch.from_numpy(np.load(controlnet_path)).to(device)

        # Compare
        base_vs_original.append(cosine_similarity(emb_base, emb_original))
        controlnet_vs_original.append(cosine_similarity(emb_controlnet, emb_original))
        valid_indices.append(sample_id)

    except Exception as e:
        print(f"Skipping {fname}: {e}")


# ========== Save and Plot Results ==========
base_vs_original = np.array(base_vs_original)
controlnet_vs_original = np.array(controlnet_vs_original)
delta = controlnet_vs_original - base_vs_original

plt.figure(figsize=(10, 6))
plt.hist(base_vs_original, bins=50, alpha=0.6, label="Base vs Original")
plt.hist(controlnet_vs_original, bins=50, alpha=0.6, label="ControlNet vs Original")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Cosine Similarity to Original Embedding")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "similarity_hist.png"))

plt.figure(figsize=(10, 6))
plt.hist(delta, bins=50, alpha=0.7, color="purple")
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.title("ControlNet - Base Cosine Similarity Delta")
plt.xlabel("Cosine Similarity Difference")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "similarity_delta_hist.png"))

print(f"\n--- Cosine Similarity Delta Stats ---")
print(f"Samples evaluated: {len(delta)}")
print(f"Mean Δ: {np.mean(delta):.4f}")
print(f"> 0 improvement: {(delta > 0).sum()} / {len(delta)}")
print(f"> 0.01 improvement: {(delta > 0.01).sum()} samples")
print(f"> 0.05 improvement: {(delta > 0.05).sum()} samples")
print(f"< 0 (worse): {(delta < 0).sum()} samples")

np.savez(os.path.join(SAVE_DIR, "cosine_similarity_results.npz"),
         base_vs_original=base_vs_original,
         controlnet_vs_original=controlnet_vs_original,
         delta=delta,
         indices=np.array(valid_indices))
