import os
import random
import torch
import numpy as np
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from backbones.iresnet import iresnet100
import torch.nn as nn
from torch.nn import functional as F

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
MASK_DIR = "./DATASET_EVAL/masks" 
EMBEDDING_PREFIX = "../../"
OUTPUT_DIR = "./comparisons_random"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = [5, 10]  
MASK_NAMES = {
    "eyes_big_vertical": "Eyes",
    "nose": "Nose",
    "mouth": "Mouth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ARCFACE SETUP ==========
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def image_to_tensor(img):
    return transform(img).unsqueeze(0)

# ========== DRAW STRIPS ==========
# ========== DRAW STRIPS ==========
def create_strip(samples, save_path):
    font_size = 48  # larger letters
    # Try to load a TTF font, fallback to default if not found
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf"
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue
    if font is None:
        print("Warning: Could not find a TTF font file, using default font.")
        font = ImageFont.load_default()

    img_w, img_h = samples[0]["original"].size

    # spacing
    x_spacing = 50  # horizontal spacing between images
    y_spacing = 100  # vertical spacing between rows

    total_w = (len(MASK_NAMES) + 1) * img_w + x_spacing * len(MASK_NAMES)
    total_h = len(samples) * img_h + y_spacing * len(samples)

    result = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(result)

    y_offset = 0
    for sample in samples:
        # paste original
        result.paste(sample["original"], (0, y_offset))

        # paste masked + similarity
        for j, (mask_key, mask_label) in enumerate(MASK_NAMES.items()):
            x = (j + 1) * img_w + j * x_spacing
            result.paste(sample["masked"][mask_key], (x, y_offset))
            sim = sample["sims"][mask_key]

            # create text
            text = f"{mask_label}: {sim:.3f}"
            # measure text size using textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # draw text centered below the image
            draw.text(
                (x + (img_w - text_w)//2, y_offset + img_h + 10),
                text,
                fill="black",
                font=font
            )

        y_offset += img_h + y_spacing

    result.save(save_path)
    print(f"Saved: {save_path}")


# ========== MAIN ==========
def main():
    arcface_model_path = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
    arcface = ARCFACE(arcface_model_path)

    dataset = load_dataset(DATASET_NAME, split="train")

    for N in SAMPLES:
        chosen = random.sample(range(2947), N)
        samples = []

        for idx in chosen:
            sample = dataset[idx]
            
            image = sample["image"].resize((512, 512))
            embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

            masked_images = {}
            sims = {}
            for mask_key in MASK_NAMES.keys():
                mask_path = os.path.join(MASK_DIR, mask_key, f"sample_{idx}.png")

                masked_img = Image.open(mask_path).convert("RGB").resize((512, 512))
                masked_images[mask_key] = masked_img

                masked_emb = arcface(image_to_tensor(masked_img).to(device)).squeeze(0)
                sims[mask_key] = cosine_similarity(embedding.squeeze(0), masked_emb)

            samples.append({
                "original": image,
                "masked": masked_images,
                "sims": sims
            })

        save_path = os.path.join(OUTPUT_DIR, f"comparison_{N}.png")
        create_strip(samples, save_path)

if __name__ == "__main__":
    main()
