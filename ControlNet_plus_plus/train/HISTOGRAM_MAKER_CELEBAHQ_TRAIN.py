import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIG --------
DATASET_NAME = "Milocas/celebahq_clean"
EMBEDDING_PREFIX = "../../"
embedding_dir_base = "./TEST_CELEBAHQ/TRAIN/embeddings_base"
embedding_dir_controlnet = "./TEST_CELEBAHQ/TRAIN/embeddings_controlnet"  
NUM_SAMPLES = 2953 
SAVE_PREFIX = "celeba-hq"  # or "ffhq"
# ------------------------

dataset = load_dataset(DATASET_NAME, split="train")

cosine_base = []
cosine_cn = []

for i in tqdm(range(NUM_SAMPLES), desc="Computing similarities"):
    sample = dataset[i]
    gt = torch.from_numpy(
        np.load(EMBEDDING_PREFIX + sample["condition"])
    ).unsqueeze(0)

    emb_base = torch.from_numpy(np.load(os.path.join(embedding_dir_base, f"{i}.npy"))).unsqueeze(0)
    emb_cn = torch.from_numpy(np.load(os.path.join(embedding_dir_controlnet, f"{i}.npy"))).unsqueeze(0)

    gt = torch.nn.functional.normalize(gt, dim=1)
    emb_base = torch.nn.functional.normalize(emb_base, dim=1)
    emb_cn = torch.nn.functional.normalize(emb_cn, dim=1)

    sim_base = torch.sum(gt * emb_base).item()
    sim_cn = torch.sum(gt * emb_cn).item()

    cosine_base.append(sim_base)
    cosine_cn.append(sim_cn)

cosine_base = np.array(cosine_base)
cosine_cn = np.array(cosine_cn)
delta = cosine_cn - cosine_base

# -------- PLOTS --------
plt.figure(figsize=(8,5))
plt.hist(cosine_base, bins=50, alpha=0.6, label="Baseline (SD)")
plt.hist(cosine_cn, bins=50, alpha=0.6, label="ID-ControlNet")
plt.xlabel("Cosine Similarity", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title(f"Cosine Similarity Distribution ({SAVE_PREFIX.upper()})", fontsize=19)
plt.legend()
plt.tight_layout()
plt.savefig(f"TEST_CELEBAHQ/TRAIN/similarity_hist_{SAVE_PREFIX}.png", dpi=300)

plt.figure(figsize=(8,5))
plt.hist(delta, bins=50, alpha=0.7, color="purple")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Delta (ControlNet − Baseline)", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title(f"Delta Histogram of Cosine Similarities ({SAVE_PREFIX.upper()})", fontsize=19)
plt.tight_layout()
plt.savefig(f"TEST_CELEBAHQ/TRAIN/similarity_delta_hist_{SAVE_PREFIX}.png", dpi=300)

# Compute table statistics
mean_delta = np.mean(delta)
pct_gt_0   = np.mean(delta > 0) * 100
pct_gt_001 = np.mean(delta > 0.01) * 100
pct_gt_005 = np.mean(delta > 0.05) * 100
pct_lt_0   = np.mean(delta < 0) * 100

# Print results
print("\n=== Identity Preservation Results ===")
print(f"Dataset: {SAVE_PREFIX}")
print(f"Mean Δ: {mean_delta:+.3f}")
print(f"% Δ > 0   : {pct_gt_0:.1f}%")
print(f"% Δ > 0.01: {pct_gt_001:.1f}%")
print(f"% Δ > 0.05: {pct_gt_005:.1f}%")
print(f"% Δ < 0   : {pct_lt_0:.1f}%")


