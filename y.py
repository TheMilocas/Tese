#!/usr/bin/env python3
"""
FID and KID evaluation script following the PVA protocol (no argparse version).

Compares a folder of generated images against a folder of
ground-truth (unmasked/original) images, computing FID and KID metrics
using the same CleanFID feature extractor configuration as PVA.
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm
from cleanfid.fid import frechet_distance, get_files_features, kernel_distance
from cleanfid.features import build_feature_extractor

# ---------------------------------------------------------------------
# CONFIGURATION (edit these paths as needed)
# ---------------------------------------------------------------------
GEN_DIR  = "PVA-CelebAHQ-IDI/results_fpip_small_1.0/Inpaint_Cross_common"
GT_DIR   = "ControlNet_plus_plus/train/COMPARE_PVA/images"
OUT_DIR  = "EVAL/fid_kid_pva"

CACHE_GT_FEATS  = "EVAL/fid_kid/gt_feats.npz"
CACHE_GEN_FEATS = None     # e.g., "EVAL/fid_kid/controlnet_feats.npz"
DEVICE = "cuda"            # or "cpu"
# ---------------------------------------------------------------------


def compute_stats(feats: np.ndarray):
    """Return mean and covariance of features."""
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def get_feats_cached(img_dir: str, model, cache_path: str = None):
    """Load cached features if available; otherwise compute and save."""
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path}")
        data = np.load(cache_path)
        feats = data["feats"]
    else:
        print(f"[INFO] Extracting features from {img_dir}")
        img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
        if len(img_files) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        feats = get_files_features(img_files, model)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, feats=feats)
    return feats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build CleanFID feature extractor
    print(f"[INFO] Building CleanFID feature extractor on {DEVICE}")
    model = build_feature_extractor(mode="clean", device=DEVICE)

    # Compute features (with caching if available)
    gt_feats = get_feats_cached(GT_DIR, model, cache_path=CACHE_GT_FEATS)
    gen_feats = get_feats_cached(GEN_DIR, model, cache_path=CACHE_GEN_FEATS)

    # Compute FID
    mu_gen, cov_gen = compute_stats(gen_feats)
    mu_gt, cov_gt = compute_stats(gt_feats)
    fid_value = frechet_distance(mu_gen, cov_gen, mu_gt, cov_gt)
    print(f"[RESULT] FID = {fid_value:.3f}")

    # Compute KID (×1000 to match PVA reporting)
    kid_value = kernel_distance(gen_feats, gt_feats) * 1000
    print(f"[RESULT] KID ×1e3 = {kid_value:.3f}")

    # Save results
    results = {
        "gen_dir": GEN_DIR,
        "gt_dir": GT_DIR,
        "FID": float(fid_value),
        "KID_x1e3": float(kid_value)
    }
    out_path = os.path.join(OUT_DIR, "fid_kid_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
