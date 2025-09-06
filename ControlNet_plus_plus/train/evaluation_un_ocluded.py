import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel

from backbones.iresnet import iresnet100
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

# ========== CONFIG ==========
CONTROLNET_PATH = "../../identity_controlnet_final"
MASKED_IMG_DIR = "./FFHQ_inputs/input"
UNMASKED_EMB_DIR = "./FFHQ_inputs/input_embeddings"
MASKED_EMB_DIR = "./FFHQ_inputs/input_embeddings_masked"
MASK_PATH = "./mask.png"
NUM_SAMPLES = 2953
SAVE_DIR = "./comparison_outputs_FFHQ_masked_vs_unmasked"

os.makedirs(SAVE_DIR, exist_ok=True)

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

# ========== Evaluation Containers ==========
results = {
    "unmasked": {
        "base": [],
        "controlnet": []
    },
    "masked": {
        "base": [],
        "controlnet": []
    }
}

valid_indices = []
mask = Image.open(MASK_PATH).resize((512, 512))

for i in tqdm(range(NUM_SAMPLES), desc="Processing"):
    try:
        name = f"{i:03d}.png"
        masked_img_path = os.path.join(MASKED_IMG_DIR, name)
        unmasked_emb_path = os.path.join(UNMASKED_EMB_DIR, f"{i:03d}.npy")
        masked_emb_path = os.path.join(MASKED_EMB_DIR, f"{i:03d}.npy")

        if not (os.path.exists(masked_img_path) and os.path.exists(unmasked_emb_path) and os.path.exists(masked_emb_path)):
            continue

        masked_img = Image.open(masked_img_path).convert("RGB")
        emb_unmasked = torch.from_numpy(np.load(unmasked_emb_path)).squeeze(0)
        emb_masked = torch.from_numpy(np.load(masked_emb_path)).squeeze(0)
        emb_unmasked_input = emb_unmasked.unsqueeze(0).to(device)
        emb_masked_input = emb_masked.unsqueeze(0).to(device)

        generator = torch.Generator(device).manual_seed(1000 + i)

        # === UNMASKED EMBEDDING ===
        with torch.no_grad(), torch.autocast(device.type):
            img_base = pipe_base(
                prompt="",
                image=masked_img,
                mask_image=mask,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

            img_ctrl = pipe_controlnet(
                prompt="",
                image=masked_img,
                mask_image=mask,
                control_image=emb_unmasked_input,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

        emb_base = compute_embedding(img_base)
        emb_ctrl = compute_embedding(img_ctrl)
        results["unmasked"]["base"].append(cosine_similarity(emb_base, emb_unmasked))
        results["unmasked"]["controlnet"].append(cosine_similarity(emb_ctrl, emb_unmasked))

        # === MASKED EMBEDDING ===
        with torch.no_grad(), torch.autocast(device.type):
            img_base_m = pipe_base(
                prompt="",
                image=masked_img,
                mask_image=mask,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

            img_ctrl_m = pipe_controlnet(
                prompt="",
                image=masked_img,
                mask_image=mask,
                control_image=emb_masked_input,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

        emb_base_m = compute_embedding(img_base_m)
        emb_ctrl_m = compute_embedding(img_ctrl_m)
        results["masked"]["base"].append(cosine_similarity(emb_base_m, emb_masked))
        results["masked"]["controlnet"].append(cosine_similarity(emb_ctrl_m, emb_masked))

        valid_indices.append(i)

    except Exception as e:
        print(f"Skipping index {i}: {e}")

# ========== Convert and Save ==========
for test_type in ["unmasked", "masked"]:
    for method in ["base", "controlnet"]:
        results[test_type][method] = np.array(results[test_type][method])

    # Save npz
    np.savez(os.path.join(SAVE_DIR, f"results_{test_type}.npz"),
             base_vs_original=results[test_type]["base"],
             controlnet_vs_original=results[test_type]["controlnet"],
             delta=results[test_type]["controlnet"] - results[test_type]["base"],
             indices=np.array(valid_indices))

# ========== Plot ==========
plt.figure(figsize=(12, 6))
for test_type in ["unmasked", "masked"]:
    plt.hist(results[test_type]["base"], bins=50, alpha=0.4, label=f"{test_type.capitalize()} - Base")
    plt.hist(results[test_type]["controlnet"], bins=50, alpha=0.4, label=f"{test_type.capitalize()} - ControlNet")

plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Base vs ControlNet - Cosine Similarity (Unmasked vs Masked Embeddings)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "masked_vs_unmasked_histogram.png"))

# ========== Print Stats ==========
for test_type in ["unmasked", "masked"]:
    delta = results[test_type]["controlnet"] - results[test_type]["base"]
    print(f"\n--- Cosine Similarity Delta Stats for {test_type.upper()} embeddings ---")
    print(f"Samples evaluated: {len(delta)}")
    print(f"Mean Δ: {np.mean(delta):.4f}")
    print(f"> 0 improvement: {(delta > 0).sum()} / {len(delta)}")
    print(f"> 0.01 improvement: {(delta > 0.01).sum()} samples")
    print(f"> 0.05 improvement: {(delta > 0.05).sum()} samples")
    print(f"< 0 (worse): {(delta < 0).sum()} samples")
