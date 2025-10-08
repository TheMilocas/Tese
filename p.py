#!/usr/bin/env python3
"""
PVA-style identity similarity evaluation (robust version with debug prints).
Run from your project root.
"""

import os
import sys
import glob
import traceback
import numpy as np
from tqdm import tqdm

# -------- CONFIG (edit these paths if needed) ----------------
GEN_DIR  = "PVA-CelebAHQ-IDI/results_fpip_small_1.0/Inpaint_Cross_common"
GT_DIR   = "ControlNet_plus_plus/train/COMPARE_PVA/images"
OUT_DIR  = "EVAL/eval_pva"
PVA_LIB  = os.path.join("PVA-CelebAHQ-IDI")   # location of PVA lib
BATCH_SZ = 32
# --------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Add absolute PVA lib path to sys.path
pva_lib_abspath = os.path.abspath(PVA_LIB)
print(f"[INFO] PVA lib path -> {pva_lib_abspath}")
if not os.path.isdir(pva_lib_abspath):
    print(f"[ERROR] PVA lib path does not exist: {pva_lib_abspath}")
    print("Make sure PVA-CelebAHQ-IDI/lib exists relative to project root.")
    sys.exit(1)
sys.path.insert(0, pva_lib_abspath)

# Try import PVA helpers
try:
    from lib.misc import imread_pil, dict_append
    from lib.face_net import IDSimilarity
    print("[INFO] Imported PVA helpers: imread_pil, IDSimilarity")
except Exception as e:
    print("[ERROR] Failed to import PVA helper modules.")
    traceback.print_exc()
    print("\nDirectory listing of PVA lib for debugging:")
    print(os.listdir(pva_lib_abspath))
    sys.exit(1)

# Helper: convert PIL returned by imread_pil -> tensor used by PVA code style
import torch
def pil2tensor_fn(pil_img):
    """
    Same mapping used in PVA: numpy->tensor, permute, scale to [-1,1]
    imread_pil already returns a PIL image sized to (256,256) if invoked that way.
    """
    arr = np.asarray(pil_img).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float().cuda() / 127.5 - 1
    return t

# Helper to batch inference (copied/compatible with PVA)
def batch_inference(func, data, batch_size=BATCH_SZ):
    res = []
    if len(data) == 0:
        return torch.empty(0)
    for i in tqdm(range(len(data) // batch_size + 1), desc="batches"):
        st, ed = i * batch_size, (i + 1) * batch_size
        st, ed = min(st, len(data)), min(ed, len(data))
        if st == ed:
            break
        res.append(func(data[st:ed]))
    if len(res) == 0:
        return torch.empty(0)
    return torch.cat(res)

# Filename parsing (your GT: {something}_{imageid}.ext, GEN: {something}_{imageid}_{maskid}.ext)
def get_image_id_from_gt(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    return parts[-1]   # last token is imageid for GT

def get_image_id_from_gen(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    # guard: if only two parts treat as imageid last part
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]

def get_mask_id_from_gen(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    # mask id typically last token for gen files (or contains 'maskX' patterns)
    last = parts[-1]
    # try parse ints directly
    try:
        return int(last)
    except:
        # try 'maskX' pattern
        import re
        m = re.search(r"mask(\d+)", last)
        if m:
            return int(m.group(1))
    return None

# Check directories & show samples
print(f"[INFO] GEN_DIR = {GEN_DIR} -> exists: {os.path.isdir(GEN_DIR)}")
print(f"[INFO] GT_DIR  = {GT_DIR}  -> exists: {os.path.isdir(GT_DIR)}")
gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "*")))
gt_files  = sorted(glob.glob(os.path.join(GT_DIR, "*")))
print(f"[INFO] Found {len(gt_files)} ground-truth files, {len(gen_files)} generated files")
if len(gt_files) == 0 or len(gen_files) == 0:
    print("[ERROR] One of the directories is empty. Check paths and filenames.")
    sys.exit(1)

print("[INFO] example GT files:", gt_files[:3])
print("[INFO] example GEN files:", gen_files[:6])

# Build GT mapping (image_id -> path)
gt_map = {}
for p in gt_files:
    imgid = get_image_id_from_gt(p)
    if imgid in gt_map:
        print(f"[WARN] duplicate GT id {imgid} -> {p}")
    gt_map[imgid] = p
print(f"[INFO] GT map built for {len(gt_map)} unique ids")

# Prepare IDSimilarity (PVA style). Use CUDA if present, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {device}")
try:
    id_crit = IDSimilarity(model_type="glint")
    # try to move to device if possible (PVA code used .cuda())
    try:
        id_crit = id_crit.to(device)
    except Exception:
        try:
            id_crit = id_crit.cuda()
        except Exception:
            pass
    print("[INFO] IDSimilarity loaded")
except Exception as e:
    print("[ERROR] Failed to instantiate IDSimilarity")
    traceback.print_exc()
    sys.exit(1)

# Compute GT features in batches (we follow PVA flow)
gt_ids = list(gt_map.keys())
gt_paths = [gt_map[i] for i in gt_ids]

def _read_batch_get_feats(paths_batch):
    imgs = [pil2tensor_fn(imread_pil(p, (256,256))).cuda() for p in paths_batch]
    x = torch.stack(imgs, dim=0)
    with torch.no_grad():
        feats = id_crit.extract_feats(x)   # returns tensor [B, D]
    return feats.cpu()

print("[INFO] Computing GT identity features (this may take a while)...")
gt_feats_list = batch_inference(_read_batch_get_feats, gt_paths, batch_size=BATCH_SZ)
# Normalize
gt_feats_list = gt_feats_list / gt_feats_list.norm(p=2, dim=1, keepdim=True)
# Map image id -> feature (cpu tensor)
id_feat_gt = {imgid: gt_feats_list[idx:idx+1] for idx, imgid in enumerate(gt_ids)}
print("[INFO] Saved GT features for %d ids" % len(id_feat_gt))

# Now compute features for all generated files (batch inference)
gen_files_sorted = sorted(gen_files)
print("[INFO] Computing generated image features...")
def _read_batch_get_feats_gen(paths_batch):
    imgs = [pil2tensor_fn(imread_pil(p, (256,256))).cuda() for p in paths_batch]
    x = torch.stack(imgs, dim=0)
    with torch.no_grad():
        feats = id_crit.extract_feats(x)
    return feats.cpu()

gen_feats = batch_inference(_read_batch_get_feats_gen, gen_files_sorted, batch_size=BATCH_SZ)
gen_feats = gen_feats / gen_feats.norm(p=2, dim=1, keepdim=True)
print(f"[INFO] Computed features for {len(gen_feats)} generated images")

# Map each generated file to its image id and mask id, and compute cosine with GT
per_image_cosines = {}   # image_id -> list of cosines (all masks)
per_mask_stats = {}      # mask_id -> list of cosines
missing_gt = 0

for idx, fpath in enumerate(gen_files_sorted):
    try:
        img_id = get_image_id_from_gen(fpath)
        mask_id = get_mask_id_from_gen(fpath)
        feat = gen_feats[idx:idx+1]   # CPU tensor (1,D)
        if img_id not in id_feat_gt:
            missing_gt += 1
            continue
        gtfeat = id_feat_gt[img_id]   # (1,D)
        cos = float((gtfeat * feat).sum().item())
        per_image_cosines.setdefault(img_id, []).append(cos)
        if mask_id is not None:
            per_mask_stats.setdefault(mask_id, []).append(cos)
    except Exception as e:
        print(f"[WARN] failed for {fpath}: {e}")
        traceback.print_exc()

if missing_gt:
    print(f"[WARN] {missing_gt} generated files had no matching GT embedding (skipped).")

# Aggregate to per-image means
per_image_mean = {imgid: float(np.mean(vals)) for imgid, vals in per_image_cosines.items()}
overall_mean = float(np.mean(list(per_image_mean.values()))) if len(per_image_mean)>0 else float("nan")

# Per-mask stats
per_mask_summary = {}
for mask_id, vals in per_mask_stats.items():
    per_mask_summary[mask_id] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "count": len(vals)}

# Summary prints
print("=== SUMMARY ===")
print(f"Images evaluated (unique ids): {len(per_image_mean)}")
print(f"Overall mean cosine similarity: {overall_mean:.4f}")
print("Per-mask summary:")
for mid, stats in sorted(per_mask_summary.items()):
    print(f"  mask {mid}: mean {stats['mean']:.4f} std {stats['std']:.4f} count {stats['count']}")

# Save outputs
np.save(os.path.join(OUT_DIR, "per_image_mean.npy"), per_image_mean)
np.save(os.path.join(OUT_DIR, "per_mask_summary.npy"), per_mask_summary)
with open(os.path.join(OUT_DIR, "overall.txt"), "w") as fh:
    fh.write(f"{overall_mean:.6f}\n")

print(f"[INFO] Results saved to {OUT_DIR}")
