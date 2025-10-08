import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
OCCLUDED_DIR = "./DATASET_EVAL/images_ocluded_table"
GEN_DIR_10 = "./COMPARE_IDS_CONTROLNET/grid_10x10" #"./COMPARE_IDS_CONTROLNET/masked_eval"
SIMS_PATH = "./COMPARE_IDS_CONTROLNET/sims_10x10.csv" #"./COMPARE_IDS_CONTROLNET/masked_eval/sims_masked_10x10.csv"

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

def load_unoccluded_imgs(dataset, N):
    return [dataset[i]["image"].convert("RGB") for i in range(N)]

def load_occluded_imgs(N, folder=OCCLUDED_DIR):
    return [Image.open(os.path.join(folder, f"{i}.png")).convert("RGB") for i in range(N)]

# ========== BUILD GRID ==========
def build_grid(unoccluded_imgs, occluded_imgs, N, gen_dir, save_img, sims_csv=None):
    sims = None
    if sims_csv and os.path.exists(sims_csv):
        sims = pd.read_csv(sims_csv, index_col=0)

    fig, axes = plt.subplots(N+1, N+1, figsize=(2*N, 2*N))

    # blank top-left corner
    axes[0, 0].axis("off")

    # column headers: unoccluded
    for j in range(N):
        axes[0, j+1].imshow(unoccluded_imgs[j])
        axes[0, j+1].axis("off")

    # row headers: occluded
    for i in range(N):
        axes[i+1, 0].imshow(occluded_imgs[i])
        axes[i+1, 0].axis("off")

    # fill grid with generated images + similarities
    for i in range(N):
        for j in range(N):
            fname = f"gen_row{i}_col{j}.png"
            path = os.path.join(gen_dir, fname)

            if os.path.exists(path):
                gen_img = Image.open(path).convert("RGB")
                axes[i+1, j+1].imshow(gen_img)
            else:
                axes[i+1, j+1].imshow(Image.new("RGB", (64, 64), (255, 255, 255)))

            axes[i+1, j+1].axis("off")

            # similarity text below image
            if sims is not None:
                sim_val = sims.iloc[i, j]
                # place text at bottom-center of the subplot
                axes[i+1, j+1].text(
                    0.5, -0.05, f"{sim_val:.4f}",
                    fontsize=12, ha="center", va="top",
                    transform=axes[i+1, j+1].transAxes
                )


    plt.tight_layout()
    plt.savefig(save_img, dpi=200)
    plt.close()

# ========== RUN ==========

# Full 10×10
unoccluded_imgs10 = load_unoccluded_imgs(dataset, 10)
occluded_imgs10 = load_occluded_imgs(10)

build_grid(unoccluded_imgs10, occluded_imgs10,
           N=10,
           gen_dir=GEN_DIR_10,
           save_img="grid10x10.png",
           sims_csv=SIMS_PATH)

# First 6×6 subset (still uses sims_10x10.csv, but crops automatically)
build_grid(unoccluded_imgs10[:6], occluded_imgs10[:6],
           N=6,
           gen_dir=GEN_DIR_10,
           save_img="grid6x6.png",
           sims_csv=SIMS_PATH)
