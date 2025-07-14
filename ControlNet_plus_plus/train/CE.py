import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F
from backbones.iresnet import iresnet100 

# ========== ARCFACE LOADING ==========
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

# ========== SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ARCFACE/models/R100_MS1MV3/backbone.pth"))
arcface = ARCFACE(arcface_model_path)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# ========== EMBEDDING LOOP ==========
def save_embeddings_all_folders(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    folder_names = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

    for folder in tqdm(folder_names, desc="Folders"):
        input_folder = os.path.join(input_root, folder)
        output_folder = os.path.join(output_root, folder)
        os.makedirs(output_folder, exist_ok=True)

        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")])

        for img_file in image_files:
            img_path = os.path.join(input_folder, img_file)
            try:
                img_tensor = preprocess_image(img_path)
                emb = arcface(img_tensor).squeeze(0).cpu().numpy()

                emb_save_path = os.path.join(output_folder, os.path.splitext(img_file)[0] + ".npy")
                np.save(emb_save_path, emb)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

# ========== PATHS ==========
input_base_dir = "./seed_randomness_outputs/base"
output_base_emb_dir = "./seed_randomness_outputs/embeddings_base"

input_controlnet_dir = "./seed_randomness_outputs/controlnet"
output_controlnet_emb_dir = "./seed_randomness_outputs/embeddings_controlnet"

# ========== RUN ==========
print("Processing Base Model Images...")
save_embeddings_all_folders(input_base_dir, output_base_emb_dir)

print("Processing ControlNet Images...")
save_embeddings_all_folders(input_controlnet_dir, output_controlnet_emb_dir)
