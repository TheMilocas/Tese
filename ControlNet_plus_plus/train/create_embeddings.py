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

def save_embeddings_for_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")]

    for img_file in tqdm(image_files):
        img_path = os.path.join(input_folder, img_file)
        img_tensor = preprocess_image(img_path)
        emb = arcface(img_tensor).squeeze(0).cpu().numpy()

        emb_save_path = os.path.join(output_folder, os.path.splitext(img_file)[0] + ".npy")
        np.save(emb_save_path, emb)

if __name__ == "__main__":
    base_dir = os.path.abspath("comparison_outputs_random_seed/base")
    controlnet_dir = os.path.abspath("comparison_outputs_random/controlnet")

    save_embeddings_for_folder(base_dir, "comparison_outputs_random_seed/embeddings_base")
    save_embeddings_for_folder(controlnet_dir, "comparison_outputs_random_seed/embeddings_controlnet")
