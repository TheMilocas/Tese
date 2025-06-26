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
masked_out_dir = "comparison_outputs_random_seed/embeddings_masked"
unmasked_out_dir = "comparison_outputs_random_seed/embeddings_unmasked"
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
                {f.split(".")[0] for f in os.listdir(masked_out_dir)} & \
                {f.split(".")[0] for f in os.listdir(unmasked_out_dir)}

    results = {
        "base_vs_original": [],
        "base_vs_occluded": [],
        "masked_vs_original": [],
        "masked_vs_occluded": [],
        "unmasked_vs_original": [],
        "unmasked_vs_occluded": [],
    }

    for sid in tqdm(sorted(valid_ids), desc=f"Processing {category or 'general'}"):
        try:
            orig = load_embedding(os.path.join(original_dir, f"{sid}.npy")).reshape(1, -1)
            occl = load_embedding(os.path.join(occluded_dir, f"{sid}.npy")).reshape(1, -1)
            base = load_embedding(os.path.join(base_dir, f"{sid}.npy")).reshape(1, -1)
            masked = load_embedding(os.path.join(masked_out_dir, f"{sid}.npy")).reshape(1, -1)
            unmasked = load_embedding(os.path.join(unmasked_out_dir, f"{sid}.npy")).reshape(1, -1)

            results["base_vs_original"].append(cosine_similarity(base, orig)[0, 0])
            results["base_vs_occluded"].append(cosine_similarity(base, occl)[0, 0])
            results["masked_vs_original"].append(cosine_similarity(masked, orig)[0, 0])
            results["masked_vs_occluded"].append(cosine_similarity(masked, occl)[0, 0])
            results["unmasked_vs_original"].append(cosine_similarity(unmasked, orig)[0, 0])
            results["unmasked_vs_occluded"].append(cosine_similarity(unmasked, occl)[0, 0])
        except Exception as e:
            print(f"Skipping {sid} in '{category or 'general'}' due to: {e}")

    all_results[category or "general"] = results

for key in results:
    results[key] = np.array(results[key])

# Now you're safe to proceed with the analysis
base = results["base_vs_occluded"]
masked = results["masked_vs_occluded"]
unmasked = results["unmasked_vs_occluded"]

base_1 = results["base_vs_original"]
masked_1 = results["masked_vs_original"]
unmasked_1 = results["unmasked_vs_original"]

win_counts = {
    "Base wins": np.sum((base > masked) & (base > unmasked)),
    "Masked wins": np.sum((masked > base) & (masked > unmasked)),
    "Unmasked wins": np.sum((unmasked > base) & (unmasked > masked)),
    "Ties": np.sum(
        (np.isclose(base, masked, atol=1e-5)) &
        (np.isclose(base, unmasked, atol=1e-5))
    )
}

print(f"\nResults for category: {category or 'general'}")
for k, v in win_counts.items():
    print(f"{k}: {v} samples")

delta_masked_vs_base = masked - base
delta_unmasked_vs_base = unmasked - base
delta_unmasked_vs_masked = unmasked - masked

delta_masked_vs_base_1 = masked - base
delta_unmasked_vs_base_1 = unmasked - base
delta_unmasked_vs_masked_1 = unmasked - masked


plt.figure(figsize=(10, 6))
plt.hist(delta_masked_vs_base, bins=50, alpha=0.6, label="Masked - Base")
plt.hist(delta_unmasked_vs_base, bins=50, alpha=0.6, label="Unmasked - Base")
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.title("Per-sample Cosine Similarity Improvement (vs. Masked Identity)")
plt.xlabel("Delta Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_similarity_vs_masked.png")

def plot_cdf(data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(len(sorted_data)) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

plt.figure(figsize=(10, 6))
plot_cdf(delta_masked_vs_base, "Masked - Base")
plot_cdf(delta_unmasked_vs_base, "Unmasked - Base")

plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Delta Cosine Similarity")
plt.ylabel("Cumulative Proportion of Samples")
plt.title("CDF of Similarity Improvements (vs. Masked Identity)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cdf_delta_vs_masked.png")

def describe_delta(name, delta):
    print(f"{name}:")
    print(f"  Mean: {np.mean(delta):.4f}")
    print(f"  >0 improvement: {(delta > 0).sum()} / {len(delta)}")
    print(f"  >0.01 improvement: {(delta > 0.01).sum()} samples")
    print(f"  >0.05 improvement: {(delta > 0.05).sum()} samples")
    print(f"  <0 (worse): {(delta < 0).sum()} samples\n")

describe_delta("Masked vs Base", delta_masked_vs_base)
describe_delta("Unmasked vs Base", delta_unmasked_vs_base)
describe_delta("Unmasked vs Masked", delta_unmasked_vs_masked)

print(f" ------------------------------------ \n")

describe_delta("Masked vs Base", delta_masked_vs_base_1)
describe_delta("Unmasked vs Base", delta_unmasked_vs_base_1)
describe_delta("Unmasked vs Masked", delta_unmasked_vs_masked_1)


# np.savez("identity_similarity_common.npz", **results)
# print("Saved similarity results to identity_similarity_common.npz")

# plt.figure(figsize=(10, 6))
# plt.hist(results["base_vs_original"], bins=50, alpha=0.6, label="Base vs Original")
# plt.hist(results["masked_vs_original"], bins=50, alpha=0.6, label="Masked vs Original")
# plt.hist(results["unmasked_vs_original"], bins=50, alpha=0.6, label="Unmasked vs Original")
# plt.xlabel("Cosine Similarity")
# plt.ylabel("Frequency")
# plt.title("Identity Similarity vs Original Embeddings")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("identity_similarity_to_original.png")
# print("Saved plot: identity_similarity_to_original.png")

# plt.figure(figsize=(10, 6))
# plt.hist(results["base_vs_masked"], bins=50, alpha=0.6, label="Base vs Masked")
# plt.hist(results["masked_vs_masked"], bins=50, alpha=0.6, label="Masked vs Masked")
# plt.hist(results["unmasked_vs_masked"], bins=50, alpha=0.6, label="Unmasked vs Masked")
# plt.xlabel("Cosine Similarity")
# plt.ylabel("Frequency")
# plt.title("Identity Similarity vs Masked Embeddings")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("identity_similarity_to_masked.png")
# print("Saved plot: identity_similarity_to_masked.png")
