import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from backbones.iresnet import iresnet100 
from torchvision import transforms

# ==== Config Paths ====
CONTROLNET_PATH = "../../identity_controlnet_final"
MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
IMG_DIR = "./temp/input_unmasked"
MASKED_IMG_PATH = "./temp/input_masked/100.png"
EMB_DIR = "./temp/input_emebdding_unmasked"
SAVE_DIR = "./temp/temp_outputs1"
MASK_DIR = "./mask_ISR.png"
BASE_IMAGE_ID = 100 
NUM_EMBEDDINGS = len([
    f for f in os.listdir(EMB_DIR)
    if f.endswith(".npy")
])

os.makedirs(SAVE_DIR, exist_ok=True)

class ARCFACE(torch.nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
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

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

def compute_embedding(img):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return arcface(tensor).squeeze(0).cpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arcface = ARCFACE(MODEL_PATH, device)

controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# ==== Load Base Image & Ground Truth ====
target_image = Image.open(os.path.join(IMG_DIR, f"{BASE_IMAGE_ID:03d}.png")).convert("RGB")
masked_image = Image.open(MASKED_IMG_PATH).convert("RGB")
mask = Image.open(MASK_DIR).resize((512, 512))

# Get ground truth embedding
embedding_target = compute_embedding(target_image)

# ==== Generate with Each Embedding ====
cosine_similarities = []
generated_paths = []

embedding_files = sorted([
    f for f in os.listdir(EMB_DIR)
    if f.endswith(".npy")
])

for fname in tqdm(embedding_files, desc="Generating images"):
    emb_path = os.path.join(EMB_DIR, fname)
    emb_index = int(os.path.splitext(fname)[0])
    emb = torch.from_numpy(np.load(emb_path)).unsqueeze(0).to(device)
    
    generator = torch.Generator(device).manual_seed(1000 + emb_index)

    with torch.no_grad(), torch.autocast(device.type):
        result = pipe(
            prompt="",
            image=masked_image,
            mask_image=mask,
            control_image=emb,
            num_inference_steps=25,
            controlnet_conditioning_scale=1.0,
            generator=generator
        ).images[0]

    # Save output
    out_path = os.path.join(SAVE_DIR, f"{emb_index:03d}.png")
    result.save(out_path)
    generated_paths.append(out_path)

    # Compute similarity to target
    emb_gen = compute_embedding(result)
    sim = F.cosine_similarity(embedding_target.view(-1), emb_gen.view(-1), dim=0).item()
    cosine_similarities.append(sim)

# ==== Analyze Results ====
best_index = int(np.argmax(cosine_similarities))
best_similarity = cosine_similarities[best_index]
best_fname = embedding_files[best_index]
is_correct = (int(os.path.splitext(best_fname)[0]) == BASE_IMAGE_ID)

print(f"\n=== RESULTS ===")
print(f"Target image: {BASE_IMAGE_ID:03d}.png")
print(f"Best match:   {best_fname}")
print(f"Similarity:   {best_similarity:.4f}")
print(f"Correct?      {'YES' if is_correct else 'NO'}")

with open(os.path.join(SAVE_DIR, "ranked.txt"), "w") as f:
    for i, sim in sorted(enumerate(cosine_similarities), key=lambda x: -x[1]):
        f.write(f"{i:03d}.png\t{sim:.4f}\n")

indices = list(range(len(cosine_similarities)))
scores = cosine_similarities

plt.figure(figsize=(10, 5))
bars = plt.bar(indices, scores, color='skyblue')

base_index = embedding_files.index(f"{BASE_IMAGE_ID}.npy")
bars[base_index].set_color('green')
bars[best_index].set_color('orange')    # highest similarity

plt.axhline(1.0, color='red', linestyle='--', label='Perfect Match (1.0)')
plt.title("Cosine Similarities to Ground Truth Identity")
plt.xlabel("Embedding Index")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1.05)
plt.xticks(indices)
plt.legend(loc='lower right')
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "similarity_plot.png")
plt.savefig(plot_path)