import os, glob, sys, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# ---- CONFIG ----
ORIGINAL_DIR = "./ControlNet_plus_plus/train/COMPARE_PVA/images"
GEN_DIR_PVA = "./PVA-CelebAHQ-IDI/results_fpip_small_1.0/Inpaint_Cross_common"
GEN_DIR_CONTROLNET = "./ControlNet_plus_plus/train/COMPARE_PVA/output_controlnet_new_common"
GEN_DIR_BASE = "./ControlNet_plus_plus/train/COMPARE_PVA/output_base_new_common"
PVA_LIB = "./PVA-CelebAHQ-IDI"

TARGETS = {"base": 0.379, "pva": 0.619, "controlnet": 0.611}
TOL = 0.03

# ---- IMPORT PVA MODULES ----
sys.path.append(PVA_LIB)
from lib.face_net import IDSimilarity

# ---- INIT MODEL ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_model = IDSimilarity(model_type="glint").to(device).eval()

# ---- FUNCTIONS ----
def compute_feat(img_path):
    """Compute normalized embedding from an image path."""
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = id_model.extract_feats(tensor)
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).to(device)
        feat = F.normalize(feat, p=2, dim=1)  # ensure unit norm
    return feat

def cosine_sim(f1, f2):
    return F.cosine_similarity(f1, f2, dim=1).item()

def mean_sim(folder, image_id, gt_feat):
    """Mean cosine sim for files matching ID in a folder."""
    files = sorted(glob.glob(os.path.join(folder, f"*{image_id}_*.jpg")))
    if not files:
        return None
    sims = []
    for f in files:
        gen_feat = compute_feat(f)
        sims.append(cosine_sim(gt_feat, gen_feat))
    return float(np.mean(sims))

# ---- MAIN LOOP ----
image_files = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.jpg")))
print(f"Found {len(image_files)} ground truth images.")

for img_path in tqdm(image_files):
    image_id = os.path.splitext(os.path.basename(img_path))[0].split("_")[-1]
    gt_feat = compute_feat(img_path)

    s_base = mean_sim(GEN_DIR_BASE, image_id, gt_feat)
    s_pva = mean_sim(GEN_DIR_PVA, image_id, gt_feat)
    s_ctrl = mean_sim(GEN_DIR_CONTROLNET, image_id, gt_feat)

    if None in [s_base, s_pva, s_ctrl]:
        continue

    # Check within tolerance
    if (
        abs(s_base - TARGETS["base"]) <= TOL and
        abs(s_pva - TARGETS["pva"]) <= TOL and
        abs(s_ctrl - TARGETS["controlnet"]) <= TOL
    ):
        print(f"\nRepresentative found: {img_path}")
        print(f"Base={s_base:.3f}, PVA={s_pva:.3f}, ControlNet={s_ctrl:.3f}")

else:
    print("No image matched targets within tolerance.")
