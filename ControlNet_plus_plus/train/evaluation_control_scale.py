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
EMB_DIR = "./FFHQ_inputs/input_embeddings"
IMG_DIR = "./FFHQ_inputs/input"
MASK_PATH = "./mask.png"
NUM_SAMPLES = 2953
SAVE_DIR = "./comparison_outputs_FFHQ_controlnet_scale_ablation"

os.makedirs(SAVE_DIR, exist_ok=True)

embedding_dir_base = os.path.join(SAVE_DIR, "embeddings_base")
os.makedirs(embedding_dir_base, exist_ok=True)

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

# ========== Evaluation ==========
controlnet_scales = [0.1, 0.3, 0.5, 0.7, 0.9]
controlnet_results = {scale: [] for scale in controlnet_scales}
base_results = []
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
        emb_original = torch.from_numpy(np.load(embedding_path)).squeeze(0)
        controlnet_input = torch.from_numpy(np.load(embedding_path)).unsqueeze(0).to(device)

        generator = torch.Generator(device).manual_seed(1000 + i)

        # Generate base image
        with torch.no_grad(), torch.autocast(device.type):
            base_img = pipe_base(
                prompt="",
                image=image,
                mask_image=mask,
                num_inference_steps=25,
                generator=generator
            ).images[0]

        emb_base = compute_embedding(base_img)
        sim_base = cosine_similarity(emb_base, emb_original)
        base_results.append(sim_base)

        np.save(os.path.join(embedding_dir_base, f"{i:03d}.npy"), emb_base.numpy())

        # Generate for each controlnet scale
        for scale in controlnet_scales:
            with torch.no_grad(), torch.autocast(device.type):
                control_img = pipe_controlnet(
                    prompt="",
                    image=image,
                    mask_image=mask,
                    control_image=controlnet_input,
                    controlnet_conditioning_scale=scale,
                    num_inference_steps=25,
                    generator=generator
                ).images[0]

            emb_control = compute_embedding(control_img)
            sim_control = cosine_similarity(emb_control, emb_original)
            controlnet_results[scale].append(sim_control)

        valid_indices.append(i)

    except Exception as e:
        print(f"Skipping index {i}: {e}")

# ========== Convert and Plot ==========
base_results = np.array(base_results)
controlnet_results = {k: np.array(v) for k, v in controlnet_results.items()}

# Plot all
plt.figure(figsize=(12, 6))
plt.hist(base_results, bins=50, alpha=0.5, label="Base", density=True)

for scale, scores in controlnet_results.items():
    plt.hist(scores, bins=50, alpha=0.5, label=f"ControlNet scale={scale}", density=True)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Cosine Similarity to Original Embedding (Varying ControlNet Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "controlnet_scale_comparison_histogram.png"))

# ========== Print & Save Stats ==========
for scale, ctrl_vals in controlnet_results.items():
    delta = ctrl_vals - base_results
    print(f"\n--- Cosine Similarity Delta Stats for ControlNet Scale {scale} ---")
    print(f"Samples evaluated: {len(delta)}")
    print(f"Mean Δ: {np.mean(delta):.4f}")
    print(f"> 0 improvement: {(delta > 0).sum()} / {len(delta)}")
    print(f"> 0.01 improvement: {(delta > 0.01).sum()} samples")
    print(f"> 0.05 improvement: {(delta > 0.05).sum()} samples")
    print(f"< 0 (worse): {(delta < 0).sum()} samples")

    np.savez(
        os.path.join(SAVE_DIR, f"cosine_similarity_results_scale_{scale}.npz"),
        base_vs_original=base_results,
        controlnet_vs_original=ctrl_vals,
        delta=delta,
        indices=np.array(valid_indices),
    )