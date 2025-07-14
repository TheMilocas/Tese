import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== CONFIG ==========
NUM_SAMPLES = 60
DATASET_NAME = "Milocas/celebahq_clean"
EMBEDDING_PREFIX = "../../"
BASE_EMB_DIR = "./seed_randomness_outputs/embeddings_base"
CONTROLNET_EMB_DIR = "./seed_randomness_outputs/embeddings_controlnet"

# ========== UTILS ==========
def cosine_similarity(a, b):
    a = a.view(-1)
    b = b.view(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def find_best_index(emb_folder, original_emb):
    best_sim = -1.0
    best_idx = -1
    for i in range(100):
        emb_path = os.path.join(emb_folder, f"{i:03d}.npy")
        if not os.path.exists(emb_path):
            continue
        emb = torch.from_numpy(np.load(emb_path))
        sim = cosine_similarity(original_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_idx

def plot_similarity_distribution(sample_index, emb_folder, original_emb, save_path="similarity_distribution_sample10.png"):
    similarities = []
    for i in range(100):
        emb_path = os.path.join(emb_folder, f"{i:03d}.npy")
        if os.path.exists(emb_path):
            emb = torch.from_numpy(np.load(emb_path))
            sim = cosine_similarity(original_emb, emb)
            similarities.append(sim)
    if not similarities:
        print(f"No embeddings found for sample {sample_index}")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(similarities, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Cosine Similarity to Original")
    plt.ylabel("Frequency")
    plt.title(f"Cosine Similarity Distribution (Sample {sample_index + 1})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved similarity distribution plot to: {save_path}")

# ========== MAIN ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")

best_base_indices = []
best_control_indices = []

for i in tqdm(range(NUM_SAMPLES), desc="Finding best matches"):
    name = f"{i:03d}"
    try:
        clean_sample = dataset[i]
        original_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, clean_sample["condition"]))).squeeze(0)

        base_emb_folder = os.path.join(BASE_EMB_DIR, name)
        control_emb_folder = os.path.join(CONTROLNET_EMB_DIR, name)

        best_base_idx = find_best_index(base_emb_folder, original_embedding)
        best_control_idx = find_best_index(control_emb_folder, original_embedding)

        best_base_indices.append(best_base_idx)
        best_control_indices.append(best_control_idx)

        # Plot for sample 10 (index 9)
        if i == 9:
            plot_similarity_distribution(i, base_emb_folder, original_embedding)

    except Exception as e:
        print(f"Error in sample {i}: {e}")
        best_base_indices.append(-1)
        best_control_indices.append(-1)

# ========== SAVE ==========
np.save("best_base_indices.npy", np.array(best_base_indices))
np.save("best_control_indices.npy", np.array(best_control_indices))
print("Saved best indices to: best_base_indices.npy and best_control_indices.npy")
