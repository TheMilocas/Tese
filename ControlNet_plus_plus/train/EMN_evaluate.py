import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
from torch.nn import functional as F
from backbones.iresnet import iresnet100

# ========== CONFIG ==========
ROOT_DIR = "EMNB_masks"
OUTPUT_DIR = "emn_masked_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASK_NAMES = ["eyes", "mouth", "nose", "background"]
IMAGE_NAME = "image.jpg"

# ========== ARCFACE SETUP ==========
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

# ========== UTILITIES ==========
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize([0.5], [0.5])
])

def read_image_tensor(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def apply_mask(img_path, mask_path):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])
    masked_image = cv2.bitwise_and(image, cv2.bitwise_not(mask_3ch))

    return cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

def create_comparison_image(original, masked_images, similarities, save_path):
    images = [original] + masked_images
    labels = ["Original", "Eyes", "Mouth", "Nose", "Background"]

    width, height = original.size
    font_size = 28
    text_height = font_size + 10
    total_height = height + text_height * 2

    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]

    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue
    if font is None:
        print("Warning: Could not find a TTF font file, using default font (size cannot be changed).")
        font = ImageFont.load_default()

    result = Image.new("RGB", (width * 5, total_height), "white")
    draw = ImageDraw.Draw(result)

    for i in range(5):
        x = i * width
        result.paste(images[i], (x, 0))

        label_y = height + 5
        if i == 0:
            draw.text((x + 10, label_y), labels[i], fill="black", font=font)
        else:
            sim_text = f"{labels[i]} cosine: {similarities[i-1]:.3f}"
            draw.text((x + 10, label_y), sim_text, fill="black", font=font)

    result.save(save_path)


def stack_all_results(input_dir, output_path="EMN_summary_stacked.png"):
    image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".png")
    ])

    if not image_files:
        print("No images found to stack.")
        return

    images = [Image.open(f) for f in image_files]

    # Assume all images are the same size
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)

    stacked_img = Image.new("RGB", (max_width, total_height), "white")

    y_offset = 0
    for img in images:
        stacked_img.paste(img, (0, y_offset))
        y_offset += img.height

    stacked_img.save(output_path)
    print(f"Saved stacked summary image to: {output_path}")

    
# ========== MAIN ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ARCFACE/models/R100_MS1MV3/backbone.pth"))
arcface = ARCFACE(arcface_model_path)

subfolders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])

for folder in tqdm(subfolders, desc="Processing"):
    folder_path = os.path.join(ROOT_DIR, folder)

    try:
        img_path = os.path.join(folder_path, IMAGE_NAME)
        original_tensor = read_image_tensor(img_path).to(device)
        original_embedding = arcface(original_tensor).squeeze(0)

        masked_pil_images = []
        similarities = []

        for mask_name in MASK_NAMES:
            mask_path = os.path.join(folder_path, f"{mask_name}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"{mask_path} not found")

            masked_rgb = apply_mask(img_path, mask_path)
            masked_pil = Image.fromarray(masked_rgb)
            masked_tensor = transform(masked_pil).unsqueeze(0).to(device)

            emb = arcface(masked_tensor).squeeze(0)
            sim = cosine_similarity(original_embedding, emb)

            masked_pil_images.append(masked_pil)
            similarities.append(sim)

        save_path = os.path.join(OUTPUT_DIR, f"{folder}.png")
        create_comparison_image(Image.open(img_path).convert("RGB"), masked_pil_images, similarities, save_path)

    except Exception as e:
        print(f"Failed on folder {folder}: {e}")
        continue

stack_all_results(OUTPUT_DIR)

print(f"\nSaved all comparison images to: {OUTPUT_DIR}")