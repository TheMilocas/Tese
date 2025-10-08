import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from cleanfid import fid
from lib.face_net import IDSimilarity  # from PVA repo (CosFace R100)

# -------------------------------
# CONFIG
# -------------------------------
GT_DIR = "./COMPARE_PVA/images"
METHOD_DIR = "./COMPARE_PVA/output_controlnet"   # <-- change to output_baseline for baseline
RESULTS_CSV = "./COMPARE_PVA/results.csv"
RESULTS_LATEX = "./COMPARE_PVA/results.tex"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Utils
# -------------------------------

def pil2tensor(pil_img):
    """Convert PIL to torch tensor in [-1, 1], 256x256."""
    arr = np.asarray(pil_img.resize((256, 256))).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1
    return t.to(DEVICE)

def load_images_as_tensor(paths):
    from PIL import Image
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(pil2tensor(img))
    return torch.stack(tensors)

def get_cosface_similarity(gt_dir, gen_dir):
    """Compute identity similarity using CosFace R100 (as in PVA)."""
    id_model = IDSimilarity(model_type="glint").to(DEVICE)
    id_model.eval()

    # sort by filename to align GT and generated
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*")))
    gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*")))
    assert len(gt_paths) == len(gen_paths), "Mismatch between GT and generated images!"

    sims = []
    batch_size = 16
    for i in tqdm(range(0, len(gt_paths), batch_size), desc="CosFace ID"):
        gt_batch = load_images_as_tensor(gt_paths[i:i+batch_size])
        gen_batch = load_images_as_tensor(gen_paths[i:i+batch_size])

        with torch.no_grad():
            gt_feats = id_model.extract_feats(gt_batch)
            gen_feats = id_model.extract_feats(gen_batch)

            gt_feats = gt_feats / gt_feats.norm(p=2, dim=1, keepdim=True)
            gen_feats = gen_feats / gen_feats.norm(p=2, dim=1, keepdim=True)

            cosims = (gt_feats * gen_feats).sum(1).cpu().numpy()
            sims.extend(cosims.tolist())

    sims = np.array(sims)
    return sims.mean(), sims.std()

def get_fid_kid(gt_dir, gen_dir):
    """Compute FID and KID using cleanfid (same as PVA)."""
    fid_val = fid.compute_fid(gen_dir, gt_dir, mode="clean")
    kid_val = fid.compute_kid(gen_dir, gt_dir, mode="clean")
    # scale KID to match PVA table
    return fid_val, kid_val * 1000

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print(f"Evaluating {METHOD_DIR} against GT {GT_DIR}")

    # ID similarity
    id_mean, id_std = get_cosface_similarity(GT_DIR, METHOD_DIR)

    # FID & KID
    fid_val, kid_val = get_fid_kid(GT_DIR, METHOD_DIR)

    print(f"\nResults for {METHOD_DIR}:")
    print(f"ID = {id_mean:.3f} ± {id_std:.3f}")
    print(f"FID = {fid_val:.2f}")
    print(f"KID = {kid_val:.3f}")

    # Save CSV
    row = {
        "method": os.path.basename(METHOD_DIR),
        "ID_mean": id_mean,
        "ID_std": id_std,
        "FID": fid_val,
        "KID": kid_val,
    }

    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(RESULTS_CSV, index=False)

    # Save LaTeX table
    with open(RESULTS_LATEX, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Method & ID $\\uparrow$ & FID $\\downarrow$ & KID $\\downarrow$ \\\\\n")
        f.write("\\hline\n")
        for _, r in df.iterrows():
            f.write(f"{r['method']} & {r['ID_mean']:.3f} $\\pm$ {r['ID_std']:.3f} "
                    f"& {r['FID']:.2f} & {r['KID']:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Evaluation results (CosFace ID, FID, KID) comparing methods on CelebAHQ-IDI.}\n")
        f.write("\\end{table}\n")

    print(f"\nSaved results to {RESULTS_CSV} and {RESULTS_LATEX}")
