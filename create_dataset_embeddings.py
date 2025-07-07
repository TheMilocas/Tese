import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from backbones.iresnet import iresnet100

class ARCFACE(torch.nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
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

model_path = "ARCFACE/models/R100_MS1MV3/backbone.pth"
arcface = ARCFACE(model_path)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(arcface.device)
    emb = arcface(img_tensor).squeeze(0).cpu().numpy()
    return emb

def process_folder(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        try:
            embedding = get_embedding(img_path)
            np.save(
                os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy'),
                embedding.astype(np.float32)
            )
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

process_folder(
    image_dir='datasets/celebahq/image',
    output_dir='datasets/celebahq/conditions_no_mask'
)

process_folder(
    image_dir='datasets/celebahq/masked_images',
    output_dir='datasets/celebahq/conditions'
)
