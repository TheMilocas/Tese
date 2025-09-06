import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from backbones.iresnet import iresnet100
from tqdm import tqdm

INPUT_DIR = "./ISR_inputs"
EMB_DIR = "./temp/1"
MASKED_IMG_DIR = "./temp/1"
EMB_MASKED_DIR = "./temp/1"
MASK_PATH = "./mask_ISR.png"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(MASKED_IMG_DIR, exist_ok=True)
os.makedirs(EMB_MASKED_DIR, exist_ok=True)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
arcface_model_path = os.path.abspath("../../ARCFACE/models/R100_MS1MV3/backbone.pth")
arcface = ARCFACE(arcface_model_path, device=device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def compute_embedding(img):
    img = img.resize((112, 112)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return arcface(tensor).squeeze(0).cpu()

mask = Image.open(MASK_PATH).convert("L").resize((512, 512))

image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])

for img_file in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, img_file)
    image = Image.open(img_path).convert("RGB")

    # ---- 1. Save original embedding ----
    emb = compute_embedding(image)
    np.save(os.path.join(EMB_DIR, img_file.replace(".png", ".npy").replace(".jpg", ".npy")), emb.numpy())

    # ---- 2. Apply mask ----
    masked_image = image.copy()
    masked_image_np = np.array(masked_image)
    mask_np = np.array(mask)

    if mask_np.ndim == 2: 
        mask_np = np.expand_dims(mask_np, axis=-1)

    mask_bin = 1 - (mask_np / 255).astype(np.uint8) 
    masked_image_np = masked_image_np * mask_bin

    masked_image = Image.fromarray(masked_image_np)
    masked_image.save(os.path.join(MASKED_IMG_DIR, img_file))

    # ---- 3. Save embedding for masked image ----
    emb_masked = compute_embedding(masked_image)
    #np.save(os.path.join(EMB_MASKED_DIR, img_file.replace(".png", ".npy").replace(".jpg", ".npy")), emb_masked.numpy())