import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CSV_PATH = "./controlnet_scale_test/controlnet_scale_results_fid.csv"
SAVE_DIR = "./controlnet_scale_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)

# Convert metrics to numeric safely
metrics = ["cosine_similarity", "FID", "masked_lpips", "mFID"]
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

# Aggregate mean + std per scale
agg = df.groupby("scale")[metrics].agg(["mean", "std"]).reset_index()

# ---------------- PLOT FUNCTION ----------------
def plot_metric(metric, ylabel, color, filename):
    plt.figure(figsize=(8,5))
    plt.plot(
        agg["scale"], agg[(metric, "mean")],
        marker="o", color=color, label=f"Mean {ylabel}"
    )
    plt.fill_between(
        agg["scale"],
        agg[(metric, "mean")] - agg[(metric, "std")],
        agg[(metric, "mean")] + agg[(metric, "std")],
        color=color, alpha=0.2, label="±1 Std. Dev."
    )
    plt.title(f"{ylabel} vs ControlNet Scale")
    plt.xlabel("ControlNet Conditioning Scale")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    
# ---------------- GENERATE PLOTS ----------------
plot_metric("cosine_similarity", "Cosine Similarity", "tab:blue", "cosine_similarity_vs_scale.png")
# plot_metric("masked_lpips", "LPIPS", "tab:green", "lpips_vs_scale.png")

print(f"Plots saved to {SAVE_DIR}")