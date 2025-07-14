import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.metrics.pairwise import cosine_similarity

def load_embedding(path):
    return np.load(path)

original_dir = "../../datasets/celebahq/conditions_no_mask"
occluded_dir = "../../datasets/celebahq/conditions"
base_dir = "comparison_outputs_random_seed/embeddings_base"
controlnet_out_dir = "comparison_outputs_random_seed/embeddings_controlnet"
mask_root = "categorized_masks_debug"

categories = ["eyes", "eyes_mouth_nose", "eyes_nose", "mouth_nose", "nose", ""]  # "" is general

all_results = {}

for category in categories:
    if category:
        mask_dir = os.path.join(mask_root, category)
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    else:
        # General: collect all .pngs from all subdirectories
        mask_files = []
        for root, _, files in os.walk(mask_root):
            for f in files:
                if f.endswith(".png"):
                    mask_files.append(f)
    sample_ids = {
        str(int(os.path.splitext(f)[0]))
        for f in mask_files
        if os.path.splitext(f)[0].isdigit()
    }

    valid_ids = sample_ids & \
                {f.split(".")[0] for f in os.listdir(original_dir)} & \
                {f.split(".")[0] for f in os.listdir(occluded_dir)} & \
                {f.split(".")[0] for f in os.listdir(base_dir)} & \
                {f.split(".")[0] for f in os.listdir(controlnet_out_dir)}

    results = {
        "base_vs_original": [],
        "controlnet_vs_original": [],
    }

    for sid in tqdm(sorted(valid_ids), desc=f"Processing {category or 'general'}"):
        try:
            orig = load_embedding(os.path.join(original_dir, f"{sid}.npy")).reshape(1, -1)
            base = load_embedding(os.path.join(base_dir, f"{sid}.npy")).reshape(1, -1)
            controlnet = load_embedding(os.path.join(controlnet_out_dir, f"{sid}.npy")).reshape(1, -1)

            results["base_vs_original"].append(cosine_similarity(base, orig)[0, 0])
            results["controlnet_vs_original"].append(cosine_similarity(controlnet, orig)[0, 0])
        except Exception as e:
            print(f"Skipping {sid} in '{category or 'general'}' due to: {e}")

    all_results[category or "general"] = results

for key in results:
    results[key] = np.array(results[key])

base = results["base_vs_original"]
controlnet = results["controlnet_vs_original"]

win_counts = {
    "Number of samples": np.sum(base==base),
    "Controlnet wins": np.sum((controlnet > base)),
}

print(f"\nResults for category: {category or 'general'}")
for k, v in win_counts.items():
    print(f"{k}: {v} samples")


def describe_delta(name, delta):
    print(f"{name}:")
    print(f"  Mean: {np.mean(delta):.4f}")
    print(f"  >0 improvement: {(delta > 0).sum()} / {len(delta)}")
    print(f"  >0.01 improvement: {(delta > 0.01).sum()} samples")
    print(f"  >0.05 improvement: {(delta > 0.05).sum()} samples")
    print(f"  <0 (worse): {(delta < 0).sum()} samples\n")
    