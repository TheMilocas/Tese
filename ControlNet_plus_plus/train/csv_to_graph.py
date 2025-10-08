import os
import pandas as pd
import matplotlib.pyplot as plt

# ========== CONFIG ==========
CSV_PATH = "DATASET_EVAL/similarities.csv" 
OUTPUT_DIR = "DATASET_EVAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD DATA ==========
print(f"Loading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

MASK_NAMES = ["eyes", "mouth", "nose"]

# ---------- Cosine Similarity Distributions ----------
plt.figure(figsize=(10,6))
for region in MASK_NAMES:
    if region in df.columns:
        plt.hist(df[region], bins=30, alpha=0.5, label=region)
plt.legend()
plt.title("Cosine Similarity Distributions by Masked Region")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "similarity_distributions.png"))
plt.close()

# ---------- Boxplot similarities ----------
plt.figure(figsize=(8,6))
df.boxplot(column=MASK_NAMES)
plt.title("Boxplot of Similarities per Region")
plt.savefig(os.path.join(OUTPUT_DIR, "similarity_boxplots.png"))
plt.close()

# ---------- Boxplot efficiencies ----------
plt.figure(figsize=(8,6))
df.boxplot(column=[f"{r}_efficiency" for r in MASK_NAMES])
plt.title("Mask Efficiency per Region (Identity Hidden / Area)")
plt.ylabel("Efficiency")
plt.savefig(os.path.join(OUTPUT_DIR, "mask_efficiency_boxplot.png"))
plt.close()

# ---------- Scatter Area vs Identity Hidden ----------
plt.figure(figsize=(8,6))
for region in MASK_NAMES:
    plt.scatter(df[f"{region}_mask_area"], df[f"{region}_identity_hidden"],
                alpha=0.5, label=region)
plt.xlabel("Mask Area (px)")
plt.ylabel("Identity Hidden (1 - cosine similarity)")
plt.title("Tradeoff: Area vs Identity Hidden")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "mask_tradeoff.png"))
plt.close()

print(f"Graphs saved in {OUTPUT_DIR}")
