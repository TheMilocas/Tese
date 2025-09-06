import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
from datasets import load_dataset
from backbones.iresnet import iresnet100  

# ========== CONFIG ==========
MODEL_PATH = "ARCFACE/models/R100_MS1MV3/backbone.pth"
IMAGE_DATASET = "Ryan-sjtu/ffhq512-caption"
MASK_DIR = "./mask.png"
EMB_SAVE_DIR = "./ControlNet_plus_plus/train/comparison_outputs_random_seed_new_FFHQ_set/input_embeddings"
IMG_SAVE_DIR = "./ControlNet_plus_plus/train/comparison_outputs_random_seed_new_FFHQ_set/input"
NUM_SAMPLES = 2953
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EMB_SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# ========== IMAGE TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

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

arcface = ARCFACE(MODEL_PATH, DEVICE)

def get_embedding_from_image(img: Image.Image):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(arcface.device)
    emb = arcface(img_tensor).squeeze(0).cpu().numpy()
    return emb

# ========== MASK APPLICATION FUNCTION ==========
def apply_mask_to_image(image: Image.Image, mask: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L")
    image_np = np.array(image)
    mask_np = np.array(mask)
    mask_binary = (mask_np > 128).astype(np.uint8)
    for c in range(3):
        image_np[..., c] = np.where(mask_binary == 1, fill_color[c], image_np[..., c])
    return Image.fromarray(image_np)

# ========== LOAD DATASETS ==========
print("Loading datasets...")
dataset_images = load_dataset(IMAGE_DATASET, split="train")

# ========== PROCESS ==========
for i in range(NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Processing sample...")

    image = dataset_images[i]["image"].resize((512, 512))
    mask = Image.open(MASK_DIR).resize((512, 512))

    image_masked = apply_mask_to_image(image, mask, fill_color=(0, 0, 0))
    
    filename = f"{i:03d}.png"
    image_masked.save(os.path.join(IMG_SAVE_DIR, filename))
    
    embedding = get_embedding_from_image(image)
    np.save(os.path.join(EMB_SAVE_DIR, f"{i:03d}.npy"), embedding)

print("\nMasked images and embeddings created.")
