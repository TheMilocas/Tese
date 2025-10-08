import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

from backbones.iresnet import iresnet100

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
IMG_DIR = "./DATASET_EVAL/images_ocluded_table"
MASK_DIR = "./DATASET_EVAL/masks_table"
NUM_SAMPLES = 10
GEN_DIR = "./COMPARE_IDS_CONTROLNET/grid_10x10"
SAVE_DIR = "./COMPARE_IDS_CONTROLNET/masked_eval"

MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ARCFACE SETUP ==========
class ARCFACE(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(device))
            return F.normalize(x, p=2, dim=1)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def compute_embedding(img: Image.Image, model: ARCFACE):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return model(tensor).squeeze(0).cpu()

arcface = ARCFACE(MODEL_PATH)

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
samples = [dataset[i] for i in range(NUM_SAMPLES)]

embeddings, occluded_imgs, masks = [], [], []
for i, sample in enumerate(samples):
    image_masked_path = os.path.join(IMG_DIR, f"{i}.png")
    mask_path = os.path.join(MASK_DIR, f"{i}.png")

    occluded_imgs.append(Image.open(image_masked_path).convert("RGB").resize((512, 512)))
    masks.append(Image.open(mask_path).convert("L").resize((512, 512)))
    embeddings.append(torch.from_numpy(np.load("../../" + sample["condition"])).squeeze(0))

embeddings = torch.stack(embeddings)

# ========== APPLY MASKS TO GENERATED IMAGES ==========
print("Applying inverse masks to generated images...")

similarities = np.zeros((NUM_SAMPLES, NUM_SAMPLES))

for i in range(NUM_SAMPLES):      
    for j in range(NUM_SAMPLES):  
        fname = f"gen_row{i}_col{j}.png"
        gen_path = os.path.join(GEN_DIR, fname)

        if not os.path.exists(gen_path):
            print(f"Missing: {gen_path}")
            continue

        gen_img = Image.open(gen_path).convert("RGB").resize((512, 512))
        mask = masks[i]

        masked_img = Image.composite(gen_img, Image.new("RGB", gen_img.size, (0, 0, 0)), mask)

        save_path = os.path.join(SAVE_DIR, fname)
        masked_img.save(save_path)

        emb_gen = compute_embedding(masked_img, arcface)

        sim = F.cosine_similarity(emb_gen, embeddings[j].cpu(), dim=0).item()
        similarities[i, j] = sim

# ========== SAVE COSINE SIMILARITIES ==========
df = pd.DataFrame(
    similarities,
    index=[f"occ_{i}" for i in range(NUM_SAMPLES)],
    columns=[f"id_{j}" for j in range(NUM_SAMPLES)]
)
csv_path = os.path.join(SAVE_DIR, f"sims_masked_{NUM_SAMPLES}x{NUM_SAMPLES}.csv")
df.to_csv(csv_path)

print(f"Saved masked images to {SAVE_DIR}")
print(f"Saved similarities to {csv_path}")