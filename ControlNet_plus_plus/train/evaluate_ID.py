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
masked_dir = "../../datasets/celebahq/conditions"
base_dir = "comparison_outputs_random_seed/embeddings_base"
masked_out_dir = "comparison_outputs_random_seed/embeddings_masked"
unmasked_out_dir = "comparison_outputs_random_seed/embeddings_unmasked"

get_ids = lambda d: {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".npy")}
common_ids = get_ids(original_dir) & get_ids(masked_dir) & get_ids(base_dir) & get_ids(masked_out_dir) & get_ids(unmasked_out_dir)
print(f"Found {len(common_ids)} common samples across all sources.")

results = {
    "base_vs_original": [],
    "base_vs_masked": [],
    "masked_vs_original": [],
    "masked_vs_masked": [],
    "unmasked_vs_original": [],
    "unmasked_vs_masked": [],
}

for sid in tqdm(sorted(common_ids)):
    try:
        orig = load_embedding(os.path.join(original_dir, f"{sid}.npy")).reshape(1, -1)
        mask = load_embedding(os.path.join(masked_dir, f"{sid}.npy")).reshape(1, -1)
        base = load_embedding(os.path.join(base_dir, f"{sid}.npy")).reshape(1, -1)
        masked = load_embedding(os.path.join(masked_out_dir, f"{sid}.npy")).reshape(1, -1)
        unmasked = load_embedding(os.path.join(unmasked_out_dir, f"{sid}.npy")).reshape(1, -1)

        results["base_vs_original"].append(cosine_similarity(base, orig)[0, 0])
        results["base_vs_masked"].append(cosine_similarity(base, mask)[0, 0])

        results["masked_vs_original"].append(cosine_similarity(masked, orig)[0, 0])
        results["masked_vs_masked"].append(cosine_similarity(masked, mask)[0, 0])

        results["unmasked_vs_original"].append(cosine_similarity(unmasked, orig)[0, 0])
        results["unmasked_vs_masked"].append(cosine_similarity(unmasked, mask)[0, 0])

    except Exception as e:
        print(f"Skipping {sid} due to error: {e}")

for key in results:
    results[key] = np.array(results[key])

base = results["base_vs_masked"]
masked = results["masked_vs_masked"]
unmasked = results["unmasked_vs_masked"]

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

print("Per-sample winner counts (vs. MASKED input):")
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
