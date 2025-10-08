#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torchvision import transforms
from backbones.iresnet import iresnet100

# ========= CONFIG =========
GEN_DIR = "./COMPARE_IDS_CONTROLNET/sims_10x10.csv"
MODEL_PATH = "ARCFACE/models/R100_MS1MV3/backbone.pth"
DATASET_NAME = "Milocas/celebahq_clean"
OUTPUT_CSV = "comparison_results.csv"

# ========== ARCFACE MODEL ==========
class ARCFACE(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.device = device

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

# ========== IMAGE TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def get_embedding_from_image(img: Image.Image, arcface: ARCFACE):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(arcface.device)
    emb = arcface(img_tensor).squeeze(0).cpu().numpy()
    return emb

# ========== MAIN PIPELINE ==========
def compare_generated_images_with_refs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arcface = ARCFACE(MODEL_PATH, device)

    # Load reference embeddings
    print("Loading reference embeddings...")
    dataset = load_dataset(DATASET_NAME, split="train")
    reference_embeddings = [np.array(emb) for emb in dataset["condition"][:10]]

    results = []

    for col in range(10):
        row_sims = []
        for row in range(10):
            filename = f"gen_row{row}_col{col}.png"
            img_path = os.path.join(GEN_DIR, filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing file: {img_path}")

            img = Image.open(img_path).convert("RGB")
            emb = get_embedding_from_image(img, arcface)

            ref_emb = reference_embeddings[row]
            sim = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
            row_sims.append(sim)

        results.append(row_sims)
        print(f"Column {col} similarities: {row_sims}")

    # Save to CSV
    np.savetxt(
        OUTPUT_CSV,
        np.array(results).T,
        delimiter=",",
        fmt="%.6f",
        header=",".join([f"col{c}" for c in range(10)]),
        comments=""
    )
    print(f"\nSaved results to {OUTPUT_CSV}")

# ========== RUN ==========
if __name__ == "__main__":
    compare_generated_images_with_refs()
