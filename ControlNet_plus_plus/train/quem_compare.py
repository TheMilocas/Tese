import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from backbones.iresnet import iresnet100

# === Config ===
MASKED_IMG_PATH = "./temp/1/111.jpg"
EMB_DIR = "./ISR_inputs/input_embedding_unmasked"
SAVE_PATH = "./temp/similarity_to_masked_me"
os.makedirs(SAVE_PATH, exist_ok=True)

MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ArcFace model ===
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

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def compute_embedding(img: Image.Image, model: ARCFACE):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return model(tensor).squeeze(0).cpu()

# === Load model and masked image embedding ===
arcface = ARCFACE(MODEL_PATH)
masked_img = Image.open(MASKED_IMG_PATH).convert("RGB")
embedding_masked = compute_embedding(masked_img, arcface)

# === Compare with all embeddings ===
cosine_similarities = {}
embedding_files = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".npy")])

for fname in embedding_files:
    emb_path = os.path.join(EMB_DIR, fname)
    emb = torch.from_numpy(np.load(emb_path)).squeeze(0)
    sim = F.cosine_similarity(embedding_masked, emb, dim=0).item()
    cosine_similarities[fname] = sim

# === Save results ===
npz_path = os.path.join(SAVE_PATH, "masked_vs_unmasked.npz")
np.savez_compressed(npz_path, **cosine_similarities)
print(f"Saved results to {npz_path}")

# # === Plot results ===
# sorted_items = sorted(cosine_similarities.items(), key=lambda x: -x[1])
# labels = [os.path.splitext(k)[0] for k, _ in sorted_items]
# values = [v for _, v in sorted_items]

# plt.figure(figsize=(max(10, len(labels) * 0.2), 5))
# if len(labels) <= 30:
#     plt.bar(labels, values, color='steelblue')
#     plt.xticks(rotation=45)
# else:
#     plt.plot(values, color='steelblue')
#     plt.xlabel("Sorted Embedding Index")

# plt.title("Cosine Similarity: Masked vs Unmasked Embeddings")
# plt.ylabel("Cosine Similarity")
# plt.ylim(0, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plot_path = os.path.join(SAVE_PATH, "similarity_plot.png")
# plt.savefig(plot_path)
# print(f"Plot saved to {plot_path}")
plt.figure(figsize=(12, 6))
sorted_items = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_items)

bar_colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

plt.bar(range(len(values)), values, tick_label=labels, color=bar_colors)
plt.xticks(rotation=90)
plt.xlabel("Unmasked Embedding Filename")
plt.ylabel("Cosine Similarity to Masked Image")
plt.title("Cosine Similarity: Masked vs Unmasked Embeddings")
plt.tight_layout()

plot_path = os.path.join(SAVE_PATH, "similarity_barplot.png")
plt.savefig(plot_path)

print(f"Saved plot to {plot_path}")