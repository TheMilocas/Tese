import os
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch.nn import functional as F
from backbones.iresnet import iresnet100
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler


# ========== CONFIG ==========
NUM_SEEDS = 100
SAMPLE_INDEX = 0
SAVE_DIR = "./SEED_RANDOMNESS_TEST"
IMG_SAVE_DIR = "./SEED_RANDOMNESS_TEST/images"
DATASET = "Milocas/celebahq_single_mask"
EMBEDDING_PREFIX = "../../"
MASK_DIR = "./mask.png"
MASKED_DIR = "../../datasets/celebahq/eyes_masked_images/0.jpg"
BASE_SEED = 1000
STEP = 10007  

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD DATASET ==========
print("Loading dataset...")
dataset = load_dataset(DATASET, split="train")

# ========== ArcFace Model ==========
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(ARCFACE, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface_model_path = os.path.abspath("../../ARCFACE/models/R100_MS1MV3/backbone.pth")
arcface = ARCFACE(arcface_model_path)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def compute_embedding(img):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return arcface(tensor).squeeze(0)

def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, 
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

similarities = []
seed_list = []

sample = dataset[0]
emb_original = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

masked_image = Image.open(MASKED_DIR).resize((512, 512))
emb_masked = compute_embedding(masked_image)

mask = Image.open(MASK_DIR).resize((512, 512))

sim_masked = cosine_similarity(emb_original, emb_masked)

for s in tqdm(range(NUM_SEEDS), desc="Generating images"):
    seed = BASE_SEED + SAMPLE_INDEX * STEP + s  
    generator = torch.Generator(device).manual_seed(seed)

    with torch.no_grad(), torch.autocast(device.type):
        result_base = pipe_base(
            prompt="",
            image=masked_image,
            mask_image=mask,
            num_inference_steps=25,
            generator=generator,
        ).images[0]

    # Save generated image
    img_path = os.path.join(IMG_SAVE_DIR, f"gen_seed_{seed}.png")
    result_base.save(img_path)

    # Compute similarity
    emb_generated = compute_embedding(result_base)
    sim_generated = cosine_similarity(emb_generated, emb_original)

    similarities.append(sim_generated)
    seed_list.append(seed)

# Save similarities to CSV
df = pd.DataFrame({
    "seed": seed_list,
    "cosine_similarity": similarities
})
csv_path = os.path.join(SAVE_DIR, "seed_similarities.csv")
df.to_csv(csv_path, index=False)

# ========== Plot results ==========
similarities = np.array(similarities)

plt.figure(figsize=(12, 6))
plt.bar(range(len(similarities)), similarities, alpha=0.7, label="Generated samples")
plt.axhline(np.mean(similarities), color="r", linestyle="--", label=f"Mean = {np.mean(similarities):.3f}")
plt.axhline(np.max(similarities), color="g", linestyle="--", label=f"Max = {np.max(similarities):.3f}")
plt.axhline(np.min(similarities), color="b", linestyle="--", label=f"Min = {np.min(similarities):.3f}")

plt.xlabel("Seed index")
plt.ylabel("Cosine Similarity (ArcFace)")
plt.title("Seed Randomness Impact (recomputed from saved images)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "seed_randomness_barplot.png"))

print(f"Mean similarity: {np.mean(similarities):.3f}, Std: {np.std(similarities):.3f}")
print(f"Saved CSV at: {csv_path}")