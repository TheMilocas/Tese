import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- CONFIG ----------------
METRICS_CSV = "./DATASET_EVAL/CELEBAHQ/celebahq_clean_vs_masked_metrics.csv"
DATASET_NAME = "Milocas/celebahq_clean"
CONTROLNET_DIR = "./comparison_outputs_random_seed/controlnet"
BASE_DIR = "./comparison_outputs_random_seed/base"
NEW_CSV_PATH = "./comparison_outputs_random_seed/results_maskbin_comparison.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset(DATASET_NAME, split="train")

# ---------------- METRICS ----------------
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
mfid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)

transform_fid = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess_fid(img):
    tensor = transform_fid(img)
    tensor = (tensor * 255).to(torch.uint8)
    return tensor.unsqueeze(0).to(DEVICE)

def preprocess_lpips(img):
    tensor = transform_fid(img).unsqueeze(0).to(DEVICE)  # LPIPS expects [B,3,H,W] in [0,1]
    return tensor

# ---------------- LOAD MASK SIZE CSV ----------------
df_metrics = pd.read_csv(METRICS_CSV)
# The CSV columns are: image_id (index of dataset), mask_area_px, relative_area
print(df_metrics.head())

# ---------------- CREATE MASK BINS ----------------
def mask_bin(area_frac):
    if area_frac <= 0.2:
        return "small"
    elif area_frac <= 0.4:
        return "medium"
    else:
        return "large"

# Use relative_area to assign bins
df_metrics['mask_bin'] = df_metrics['relative_area'].apply(mask_bin)
print(df_metrics['mask_bin'].value_counts())

# ---------------- FUNCTION TO LOAD GENERATED IMAGE ----------------
def load_generated_image(folder, image_id):

    filename = f"sample_{image_id:03d}.png"  # zero-padded to 3 digits
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


# ---------------- COMPUTE METRICS PER BIN ----------------
results = []

for bin_name in ['small', 'medium', 'large']:
    df_bin = df_metrics[df_metrics['mask_bin'] == bin_name]
    print(f"Processing {bin_name} masks: {len(df_bin)} samples")

    for method_name, folder in [('ControlNet', CONTROLNET_DIR), ('Baseline', BASE_DIR)]:

        fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
        mfid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)
        
        gen_tensors, orig_tensors, masked_gen_tensors, masked_orig_tensors = [], [], [], []
        lpips_scores = []

        for _, row in df_bin.iterrows():
            idx = int(row['image_id'])  # use image_id from CSV
            orig_img = dataset[idx]['image'].convert("RGB")
            mask_img = dataset[idx]['mask'].convert("L").resize((256,256))
            mask_tensor = (transform_fid(mask_img) > 0.5).to(DEVICE)
        
            img_name = str(row['image_id'])  # folder name for generated images
            gen_img = load_generated_image(folder, int(row['image_id']))
            if gen_img is None:
                continue

            # FID/mFID preprocessing
            gen_tensor = preprocess_fid(gen_img)
            orig_tensor = preprocess_fid(orig_img)

            gen_tensors.append(gen_tensor)
            orig_tensors.append(orig_tensor)
            masked_gen_tensors.append(gen_tensor * mask_tensor)
            masked_orig_tensors.append(orig_tensor * mask_tensor)

            # LPIPS
            gen_lpips = preprocess_lpips(gen_img)
            orig_lpips = preprocess_lpips(orig_img)
            with torch.no_grad():
                score = lpips_metric(gen_lpips, orig_lpips).item()
            lpips_scores.append(score)

        if len(gen_tensors) < 2:
            print(f"Skipping {bin_name} - {method_name} (too few samples)")
            results.append({'mask_bin': bin_name, 'method': method_name, 'FID': None, 'mFID': None, 'LPIPS': None})
            continue

        # Concatenate batches for FID
        gen_batch = torch.cat(gen_tensors, dim=0)
        orig_batch = torch.cat(orig_tensors, dim=0)
        masked_gen_batch = torch.cat(masked_gen_tensors, dim=0)
        masked_orig_batch = torch.cat(masked_orig_tensors, dim=0)

        # Global FID
        fid_metric.reset()
        fid_metric.update(gen_batch, real=False)
        fid_metric.update(orig_batch, real=True)
        fid_score = fid_metric.compute().item()

        # Masked FID
        mfid_metric.reset()
        mfid_metric.update(masked_gen_batch, real=False)
        mfid_metric.update(masked_orig_batch, real=True)
        mfid_score = mfid_metric.compute().item()

        # LPIPS
        lpips_mean = np.mean(lpips_scores)

        results.append({
            'mask_bin': bin_name,
            'method': method_name,
            'FID': fid_score,
            'mFID': mfid_score,
            'LPIPS': lpips_mean
        })

# ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(results)
results_df.to_csv(NEW_CSV_PATH, index=False)
print(f"Saved results CSV to {NEW_CSV_PATH}")

# ---------------- VISUALIZATION ----------------
sns.set(style="whitegrid")
for metric in ['FID', 'mFID', 'LPIPS']:
    plt.figure(figsize=(8,6))
    sns.barplot(data=results_df, x='mask_bin', y=metric, hue='method')
    plt.title(f"{metric} per mask-size bin")
    plt.ylabel(metric)
    plt.xlabel("Mask Size Bin")
    plt.savefig(os.path.join(os.path.dirname(NEW_CSV_PATH), f"{metric}_maskbin_comparison.png"))
    plt.close()