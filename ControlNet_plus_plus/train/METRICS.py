import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
CSV_PATH = "./controlnet_scale_test/controlnet_scale_results.csv"
IMG_DIR = "./controlnet_scale_test/images"
DATASET_NAME = "Milocas/celebahq_clean"
NEW_CSV_PATH = "./controlnet_scale_test/controlnet_scale_results_fid.csv"
MASK_PATH = "./mask.png"  # Mask for mFID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

# ---------------- SETUP METRICS ----------------
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
mfid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)

transform_fid = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess_fid(img):
    tensor = transform_fid(img)             # float in [0,1]
    tensor = (tensor * 255).to(torch.uint8) # uint8 in [0,255]
    return tensor.unsqueeze(0).to(DEVICE)

# Load mask and convert to boolean tensor
mask_img = Image.open(MASK_PATH).convert("L").resize((256,256))
mask_tensor = (transforms.ToTensor()(mask_img) > 0.5).to(DEVICE)  # [1,H,W]

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)

# ---------------- COMPUTE FID AND MASKED FID ----------------
for scale in sorted(df['scale'].unique()):
    df_scale = df[df['scale'] == scale]

    gen_tensors, orig_tensors = [], []

    for idx, row in df_scale.iterrows():
        name = row['image']
        seed = row['seed']

        gen_path = os.path.join(IMG_DIR, f"{name}/scale_{scale:.2f}", f"seed_{seed}.png")
        if not os.path.exists(gen_path):
            continue

        # Load images
        gen_img = Image.open(gen_path).convert("RGB")
        orig_img = dataset[idx]["image"].convert("RGB")

        # Preprocess for FID
        gen_tensors.append(preprocess_fid(gen_img))
        orig_tensors.append(preprocess_fid(orig_img))

    if len(gen_tensors) < 2:
        print(f"Skipping scale {scale} (less than 2 valid samples)")
        df.loc[df['scale']==scale, 'FID'] = None
        df.loc[df['scale']==scale, 'mFID'] = None
        continue

    # Concatenate batches
    gen_batch = torch.cat(gen_tensors, dim=0)
    orig_batch = torch.cat(orig_tensors, dim=0)

    # Compute global FID
    fid_metric.reset()
    fid_metric.update(gen_batch, real=False)
    fid_metric.update(orig_batch, real=True)
    fid_score = fid_metric.compute().item()

    # Compute masked FID (mFID) using the mask
    masked_gen_batch = gen_batch * mask_tensor  
    masked_orig_batch = orig_batch * mask_tensor

    mfid_metric.reset()
    mfid_metric.update(masked_gen_batch, real=False)
    mfid_metric.update(masked_orig_batch, real=True)
    mfid_score = mfid_metric.compute().item()

    # Assign scores to all rows of this scale
    df.loc[df['scale']==scale, 'FID'] = fid_score
    df.loc[df['scale']==scale, 'mFID'] = mfid_score

# Drop old masked_ssim column if exists
df = df.drop(columns=['masked_ssim'], errors='ignore')
df.to_csv(NEW_CSV_PATH, index=False)
print(f"Saved updated CSV to {NEW_CSV_PATH}")

# ---------------- PLOTS ----------------
sns.set(style="whitegrid")
metrics = ['cosine_similarity', 'masked_lpips', 'FID', 'mFID']

for metric in metrics:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='scale', y=metric)
    sns.swarmplot(data=df, x='scale', y=metric, color=".25", alpha=0.5)
    plt.title(f"{metric} distribution across ControlNet scales")
    plt.xlabel("ControlNet Conditioning Scale")
    plt.ylabel(metric)
    plt.savefig(os.path.join(os.path.dirname(NEW_CSV_PATH), f"{metric}_vs_scale_boxplot.png"))
    plt.close()

print("Plots saved. Updated metrics computed successfully.")
