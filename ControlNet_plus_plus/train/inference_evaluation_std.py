import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

# Diffusion and ControlNet
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel

# ArcFace
from backbones.iresnet import iresnet100
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

# ========== CONFIG ==========
CONTROLNET_PATH = "../../identity_controlnet_single_mask"
EMB_DIR = "./FFHQ_inputs/input_embeddings"
IMG_DIR = "./FFHQ_inputs/input"
MASK_PATH = "./mask.png"
NUM_SAMPLES = 2953
SAVE_DIR = "./comparison_outputs_FFHQ_randomness_v2_std_CS1"

os.makedirs(SAVE_DIR, exist_ok=True)

# Embedding Directories
embedding_dir_base = os.path.join(SAVE_DIR, "embeddings_base")
embedding_dir_controlnet = os.path.join(SAVE_DIR, "embeddings_controlnet")
os.makedirs(embedding_dir_base, exist_ok=True)
os.makedirs(embedding_dir_controlnet, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return arcface(tensor).squeeze(0).cpu()

# ========== Cosine Similarity ==========
def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

# ========== Load Pipelines ==========
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# ========== Evaluation ==========
base_vs_original = []
controlnet_vs_original = []
valid_indices = []

mask = Image.open(MASK_PATH).resize((512, 512))

for i in tqdm(range(NUM_SAMPLES), desc="Processing"):
    try:
        name = f"{i:03d}.png"
        image_path = os.path.join(IMG_DIR, name)
        embedding_path = os.path.join(EMB_DIR, f"{i:03d}.npy")
        
        if not os.path.exists(image_path) or not os.path.exists(embedding_path):
            continue

        image = Image.open(image_path).convert("RGB")
        embedding_controlnet_input = torch.from_numpy(np.load(embedding_path)).unsqueeze(0).to(device)

        # Generate images
        with torch.no_grad(), torch.autocast(device.type):
            base_img = pipe_base(prompt="", 
                                 image=image, 
                                 mask_image=mask,
                                 num_inference_steps=25, 
                                 generator=torch.Generator(device).manual_seed(1000+i)).images[0]

            controlnet_img = pipe_controlnet(prompt="", 
                                             image=image, 
                                             mask_image=mask, 
                                             control_image=embedding_controlnet_input,
                                             controlnet_conditioning_scale=1.0,
                                             num_inference_steps=25, 
                                             generator=torch.Generator(device).manual_seed(1000+i)).images[0]

        # Compute embeddings
        emb_base = compute_embedding(base_img)
        emb_controlnet = compute_embedding(controlnet_img)
        emb_original = torch.from_numpy(np.load(embedding_path)).squeeze(0)

        # Save computed embeddings
        np.save(os.path.join(embedding_dir_base, f"{i:03d}.npy"), emb_base.numpy())
        np.save(os.path.join(embedding_dir_controlnet, f"{i:03d}.npy"), emb_controlnet.numpy())

        # Compare
        base_vs_original.append(cosine_similarity(emb_base, emb_original))
        controlnet_vs_original.append(cosine_similarity(emb_controlnet, emb_original))
        valid_indices.append(i)

    except Exception as e:
        print(f"Skipping index {i}: {e}")

# ========== Save and Plot Results ==========
base_vs_original = np.array(base_vs_original)
controlnet_vs_original = np.array(controlnet_vs_original)
delta = controlnet_vs_original - base_vs_original

plt.figure(figsize=(10, 6))
plt.hist(base_vs_original, bins=50, alpha=0.6, label="Base vs Original")
plt.hist(controlnet_vs_original, bins=50, alpha=0.6, label="ControlNet vs Original")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Cosine Similarity to Original Embedding")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "similarity_hist.png"))

plt.figure(figsize=(10, 6))
plt.hist(delta, bins=50, alpha=0.7, color="purple")
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.title("ControlNet - Base Cosine Similarity Delta")
plt.xlabel("Cosine Similarity Difference")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "similarity_delta_hist.png"))

print(f"\n--- Cosine Similarity Delta Stats ---")
print(f"Samples evaluated: {len(delta)}")
print(f"Mean Δ: {np.mean(delta):.4f}")
print(f"> 0 improvement: {(delta > 0).sum()} / {len(delta)}")
print(f"> 0.01 improvement: {(delta > 0.01).sum()} samples")
print(f"> 0.05 improvement: {(delta > 0.05).sum()} samples")
print(f"< 0 (worse): {(delta < 0).sum()} samples")

np.savez(os.path.join(SAVE_DIR, "cosine_similarity_results.npz"),
         base_vs_original=base_vs_original,
         controlnet_vs_original=controlnet_vs_original,
         delta=delta,
         indices=np.array(valid_indices))
