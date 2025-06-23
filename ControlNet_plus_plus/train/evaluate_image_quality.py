import os
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from glob import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
DATASET_NAME = "Milocas/celebahq_clean"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "comparison_outputs")

MODEL_DIRS = {
    "stable_diffusion": os.path.join(SAVE_DIR, "base"),
    "controlnet_masked": os.path.join(SAVE_DIR, "masked"),
    "controlnet_clean": os.path.join(SAVE_DIR, "unmasked"),
}

def get_image_file_list(folder, extensions={".png", ".jpg", ".jpeg"}):
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[-1].lower() in extensions
    ])
    return files

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_image(img):
    """Accept either a file path or a PIL.Image/Image feature from HF datasets."""
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif hasattr(img, "convert"):  # PIL.Image or HF Image feature
        img = img.convert("RGB")
    else:
        raise ValueError("Invalid image type passed to load_image.")
    return transform(img).unsqueeze(0)


def evaluate_folder(model_type, generated_dir, reference_dataset):
    print(f"\nEvaluating: {model_type}")
    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)
    kid = KernelInceptionDistance(subset_size=100, normalize=True).to(DEVICE)

    image_files = get_image_file_list(generated_dir)
    print(f"Found {len(image_files)} images in {generated_dir}")

    for idx, filename in tqdm(enumerate(image_files), total=len(image_files)):
        gen_image_path = os.path.join(generated_dir, filename)

        try:
            sample = reference_dataset[idx]
        except IndexError:
            print(f"Index {idx} out of bounds in reference dataset, skipping...")
            continue

        real_image_path = sample["image"]
        real_image = load_image(real_image_path).to(DEVICE)
        generated_image = load_image(gen_image_path).to(DEVICE)

        fid.update(generated_image, real=False)
        fid.update(real_image, real=True)
        kid.update(generated_image, real=False)
        kid.update(real_image, real=True)

    print(f"[{model_type}] FID: {fid.compute().item():.4f}")
    print(f"[{model_type}] KID: {kid.compute()[0].item():.4f}")

if __name__ == "__main__":
    reference_dataset = load_dataset(DATASET_NAME, split="test")

    for model_type, dir_path in MODEL_DIRS.items():
        evaluate_folder(model_type, dir_path, reference_dataset)
