import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from backbones.iresnet import iresnet100
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "./SAME_ID/images" 
mask_path = "mask.png" 
arcface_model_path = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
CONTROLNET_PATH = "../../identity_controlnet_face_specific"
output_dir = "./SAME_ID"
os.makedirs(output_dir, exist_ok=True)
# ----------------------------------------

# -------- ArcFace Model --------
class ARCFACE(torch.nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface = ARCFACE(arcface_model_path, device=device)

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# -------- Load embeddings from 2-6.jpg --------
embeddings = []
for i in range(2, 7):
    img_path = os.path.join(data_dir, f"{i}.jpg")
    img_tensor = preprocess_image(img_path)
    emb = arcface(img_tensor).squeeze(0)  # torch tensor
    embeddings.append(emb)

# -------- Load target image and mask --------
target_image_path = os.path.join(data_dir, "1.jpg")
target_image = Image.open(target_image_path).convert("RGB").resize((512, 512))
mask = Image.open(mask_path).resize((512, 512))

# -------- Pipeline setup --------
print("Loading ControlNet Masked Embeddings")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, safety_checker = None, requires_safety_checker = False

).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

generator = torch.Generator(device).manual_seed(2023)

# -------- Generate images and compute cosine similarity --------
def compute_image_embedding(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    return arcface(img_tensor).squeeze(0)

target_emb = compute_image_embedding(target_image)

cosine_similarities = []

# Compute cosine similarities between target image and the source images (2-6.jpg)
print("\n=== Cosine similarity between target image and source images ===")
for idx, emb in enumerate(embeddings, start=2):
    cosine = F.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
    print(f"Target vs image {idx}: Cosine similarity = {cosine:.4f}")

for idx, emb in enumerate(embeddings, start=2):
    emb_input = emb.unsqueeze(0)
    
    with torch.no_grad(), torch.autocast(device.type):
        result_image = pipe_controlnet(
            prompt="",
            image=target_image,
            mask_image=mask,
            control_image=emb_input,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
    
    gen_path = os.path.join(output_dir, f"generated_{idx}.png")
    result_image.save(gen_path)
    
    gen_emb = compute_image_embedding(result_image)
    cosine = F.cosine_similarity(gen_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
    cosine_similarities.append((idx, cosine))
    print(f"Embedding {idx} → Cosine similarity: {cosine:.4f}")

# -------- Load base Stable Diffusion inpainting pipeline --------
print("Loading base SD inpaint pipeline...")
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

# Generate baseline (no ArcFace conditioning)
with torch.no_grad(), torch.autocast(device.type):
    result_base = pipe_base(
        prompt="",
        image=target_image,
        mask_image=mask,
        num_inference_steps=25,
        generator=torch.Generator(device).manual_seed(1000),
    ).images[0]

base_path = os.path.join(output_dir, "generated_base.png")
result_base.save(base_path)

# Compute cosine similarity for baseline
base_emb = compute_image_embedding(result_base)
base_cosine = F.cosine_similarity(base_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
print(f"Base SD (no condition) → Cosine similarity: {base_cosine:.4f}")

# -------- Summary --------
print("\n=== Summary ===")
for idx, cosine in cosine_similarities:
    print(f"Embedding {idx}: Cosine similarity = {cosine:.4f}")

mean_cosine = np.mean([c for _, c in cosine_similarities])
print(f"Mean cosine similarity over 5 embeddings: {mean_cosine:.4f}")

def build_identity_grid(source_imgs, target_img, generated_imgs, source_sims, gen_sims, base_img, base_sim, save_path):
    """
    Adds a baseline (no condition) column to the left.
    """
    N = len(source_imgs)
    fig, axes = plt.subplots(2, N+1, figsize=(3*(N+1), 6))

    # Row 0: "no condition" + source images
    axes[0, 0].imshow(Image.new("RGB", (128, 128), (255, 255, 255)))  # blank white box
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.5, 0.5, "no\ncondition",
        fontsize=12, ha="center", va="center",
        transform=axes[0, 0].transAxes
    )

    for j in range(N):
        axes[0, j+1].imshow(source_imgs[j])
        axes[0, j+1].axis("off")
        axes[0, j+1].text(
            0.5, -0.05, f"{source_sims[j]:.4f}",
            fontsize=12, ha="center", va="top",
            transform=axes[0, j+1].transAxes
        )

    # Row 1: baseline + generated images
    axes[1, 0].imshow(base_img)
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, -0.05, f"{base_sim:.4f}",
        fontsize=12, ha="center", va="top",
        transform=axes[1, 0].transAxes
    )

    for j in range(N):
        axes[1, j+1].imshow(generated_imgs[j])
        axes[1, j+1].axis("off")
        axes[1, j+1].text(
            0.5, -0.05, f"{gen_sims[j]:.4f}",
            fontsize=12, ha="center", va="top",
            transform=axes[1, j+1].transAxes
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved identity grid with baseline to {save_path}")


# -------- Prepare images and similarities --------
source_images = [Image.open(os.path.join(data_dir, f"{i}.jpg")).convert("RGB").resize((128, 128)) for i in range(2,7)]
target_occluded = target_image.resize((128,128))
generated_images = [Image.open(os.path.join(output_dir, f"generated_{i}.png")).convert("RGB").resize((128,128)) for i in range(2,7)]

# Cosine similarities target vs source images
source_sims_vals = [F.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item() for emb in embeddings]

# Cosine similarities target vs generated images
gen_sims_vals = [cos for _, cos in cosine_similarities]

# -------- Build and save grid with baseline --------
grid_save_path = os.path.join(output_dir, "identity_grid_with_base.png")
build_identity_grid(
    source_images, target_occluded, generated_images,
    source_sims_vals, gen_sims_vals,
    base_img=result_base.resize((128,128)), base_sim=base_cosine,
    save_path=grid_save_path
)