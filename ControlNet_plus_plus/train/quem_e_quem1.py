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

# ==== Config ====
CONTROLNET_PATH = "../../identity_controlnet_final"
MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
IMG_DIR = "./temp/input_unmasked"
MASKED_IMG_PATH = "./temp/input_masked/100.png"
EMB_DIR = "./ISR_inputs/input_emebdding_unmasked"  #./temp/input_emebdding_unmasked
SAVE_DIR = "./temp/temp_outputs_all_ISR_w_prompt_cs05"
MASK_DIR = "./mask_ISR.png"
BASE_IMAGE_ID = 100 
N_SEEDS = 5

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

def cosine_similarity(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

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

target_image = Image.open(os.path.join(IMG_DIR, f"{BASE_IMAGE_ID:03d}.png")).convert("RGB")
masked_image = Image.open(MASKED_IMG_PATH).convert("RGB")
mask = Image.open(MASK_DIR).resize((512, 512))
embedding_target = compute_embedding(target_image)

embedding_files = sorted([
    f for f in os.listdir(EMB_DIR)
    if f.endswith(".npy")
])

cosine_similarities = []
average_similarities = []
best_images = []
generated_paths = []

for fname in tqdm(embedding_files, desc="Generating best-of-5"):
    emb_index = int(os.path.splitext(fname)[0])
    emb_path = os.path.join(EMB_DIR, fname)
    emb = torch.from_numpy(np.load(emb_path)).unsqueeze(0).to(device)

    sims = []
    best_sim = -1
    best_image = None

    for s in range(N_SEEDS):
        seed = 1000 + emb_index * 10 + s
        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad(), torch.autocast(device.type):
            result = pipe(
                prompt="realistic eyes with no glasses",
                image=masked_image,
                mask_image=mask,
                control_image=emb,
                num_inference_steps=25,
                controlnet_conditioning_scale=0.5,
                generator=generator
            ).images[0]

        emb_gen = compute_embedding(result)
        sim = cosine_similarity(embedding_target, emb_gen)
        sims.append(sim)

        if sim > best_sim:
            best_sim = sim
            best_image = result

    best_images.append(best_image)
    cosine_similarities.append(best_sim)
    average_similarities.append(np.mean(sims))

    # Save best image
    out_path = os.path.join(SAVE_DIR, f"{emb_index:03d}.png")
    best_image.save(out_path)
    generated_paths.append(out_path)

# ==== Analyze ====
best_index = int(np.argmax(cosine_similarities))
best_similarity = cosine_similarities[best_index]
best_fname = embedding_files[best_index]
is_correct = (int(os.path.splitext(best_fname)[0]) == BASE_IMAGE_ID)

print(f"\n=== RESULTS ===")
print(f"Target image: {BASE_IMAGE_ID:03d}.png")
print(f"Best match:   {best_fname}")
print(f"Similarity:   {best_similarity:.4f}")
print(f"Correct?      {'YES' if is_correct else 'NO'}")

# ==== Save ranked similarities ====
with open(os.path.join(SAVE_DIR, "ranked_best.txt"), "w") as f:
    for i, sim in sorted(enumerate(cosine_similarities), key=lambda x: -x[1]):
        emb_name = os.path.splitext(embedding_files[i])[0]
        f.write(f"{emb_name}.png\tBEST\t{sim:.4f}\n")

with open(os.path.join(SAVE_DIR, "ranked_avg.txt"), "w") as f:
    for i, sim in sorted(enumerate(average_similarities), key=lambda x: -x[1]):
        emb_name = os.path.splitext(embedding_files[i])[0]
        f.write(f"{emb_name}.png\tAVG\t{sim:.4f}\n")

# ==== Plot ====
indices = list(range(len(cosine_similarities)))
labels = [os.path.splitext(f)[0] for f in embedding_files]

plt.figure(figsize=(10, 5))
bars = plt.bar(indices, cosine_similarities, color='skyblue')

base_index = embedding_files.index(f"{BASE_IMAGE_ID}.npy")
bars[base_index].set_color('green')  # ground truth
bars[best_index].set_color('orange') # best match

plt.axhline(1.0, color='red', linestyle='--', label='Perfect Match (1.0)')
plt.title("Cosine Similarities to Ground Truth Identity")
plt.xlabel("Embedding Index")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1.05)
plt.xticks(indices, labels)
plt.legend(loc='lower right')
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "similarity_plot.png")
plt.savefig(plot_path)

import seaborn as sns

# ---- TOP-K BAR PLOT ----
TOP_K = 20
top_indices = np.argsort(cosine_similarities)[-TOP_K:][::-1]
top_labels = [os.path.splitext(embedding_files[i])[0] for i in top_indices]
top_scores = [cosine_similarities[i] for i in top_indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=top_labels, y=top_scores, palette="Blues_d")
plt.title(f"Top {TOP_K} Embeddings - Best Cosine Similarity")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "top_best_barplot.png"))

# ---- FULL LINE PLOT ----
sorted_best = sorted(cosine_similarities, reverse=True)
sorted_avg = sorted(average_similarities, reverse=True)

plt.figure(figsize=(12, 6))
plt.plot(sorted_best, label="Best of 5", color="orange")
plt.plot(sorted_avg, label="Average of 5", color="blue")
plt.axhline(1.0, linestyle='--', color='red', label="Perfect Similarity (1.0)")
plt.title("Cosine Similarity Rankings Across All Embeddings")
plt.ylabel("Cosine Similarity")
plt.xlabel("Sorted Embedding Index")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "full_sorted_lineplot.png"))
