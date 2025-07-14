import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # For headless environments

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_masked"
CLEAN_DATASET_NAME = "Milocas/celebahq_clean"
SAVE_DIR = "./comparison_outputs_random_seed_train_set"
EMBEDDING_PREFIX = "../../"

base_embedding_dir = os.path.join(SAVE_DIR, "embeddings_base")
controlnet_embedding_dir = os.path.join(SAVE_DIR, "embeddings_controlnet")

# ========== LOAD DATA ==========
print("Loading datasets...")
dataset = load_dataset(DATASET_NAME, split="train")
clean_dataset = load_dataset(CLEAN_DATASET_NAME, split="train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== METRICS ==========
def cosine_similarity(a, b):
    a = a.view(-1)
    b = b.view(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

base_vs_original = []
controlnet_vs_original = []
valid_indices = []

num_samples = len([f for f in os.listdir(base_embedding_dir) if os.path.isfile(os.path.join(base_embedding_dir, f))])
# num_samples = min(len(dataset), len(clean_dataset))

for i in tqdm(range(num_samples), desc="Evaluating full dataset"):
    try:
        sample = dataset[i]
        clean_sample = clean_dataset[i]

        original_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, clean_sample["condition"]))).squeeze(0)
        base_embedding = torch.from_numpy(np.load(os.path.join(base_embedding_dir, f"{i:03d}.npy"))).squeeze(0)
        controlnet_embedding = torch.from_numpy(np.load(os.path.join(controlnet_embedding_dir, f"{i:03d}.npy"))).squeeze(0)

        base_vs_original.append(cosine_similarity(base_embedding, original_embedding))
        controlnet_vs_original.append(cosine_similarity(controlnet_embedding, original_embedding))
        valid_indices.append(i)
    except Exception as e:
        print(f"Skipping index {i}: {e}")

base_vs_original = np.array(base_vs_original)
controlnet_vs_original = np.array(controlnet_vs_original)

# ========== PLOTS ==========

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(base_vs_original, bins=50, alpha=0.6, label="Base vs Original")
plt.hist(controlnet_vs_original, bins=50, alpha=0.6, label="ControlNet vs Original")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Cosine Similarity to Original Embedding (Full Dataset)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("similarity_hist_train.png")

# CDF
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(len(sorted_data)) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

plt.figure(figsize=(10, 6))
plot_cdf(base_vs_original, "Base vs Original")
plot_cdf(controlnet_vs_original, "ControlNet vs Original")
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Cosine Similarity")
plt.ylabel("Cumulative Proportion")
plt.title("CDF of Cosine Similarity (Full Dataset)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("similarity_cdf_train.png")

# Delta Histogram
delta = controlnet_vs_original - base_vs_original

plt.figure(figsize=(10, 6))
plt.hist(delta, bins=50, alpha=0.7, color="purple")
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.title("ControlNet - Base Cosine Similarity Delta (Full Dataset)")
plt.xlabel("Cosine Similarity Difference")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("similarity_delta_hist_train.png")

# Descriptive Statistics
print(f"\n--- Cosine Similarity Delta Stats (ControlNet - Base) ---")
print(f"Samples evaluated: {len(delta)}")
print(f"Mean Δ: {np.mean(delta):.4f}")
print(f"> 0 improvement: {(delta > 0).sum()} / {len(delta)}")
print(f"> 0.01 improvement: {(delta > 0.01).sum()} samples")
print(f"> 0.05 improvement: {(delta > 0.05).sum()} samples")
print(f"< 0 (worse): {(delta < 0).sum()} samples")

# Save raw data
np.savez("cosine_similarity_results_full_train.npz",
         base_vs_original=base_vs_original,
         controlnet_vs_original=controlnet_vs_original,
         delta=delta,
         indices=np.array(valid_indices))

print("\nFull dataset evaluation complete. Graphs and metrics saved.")
