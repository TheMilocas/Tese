import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
matplotlib.use("Agg")  # For headless environments

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_masked"
CLEAN_DATASET_NAME = "Milocas/celebahq_clean"
SAVE_DIR = "./seed_randomness_outputs"
EMBEDDING_PREFIX = "../../"

base_dir = os.path.join(SAVE_DIR, "base")
controlnet_dir = os.path.join(SAVE_DIR, "controlnet")
input_dir = os.path.join(SAVE_DIR, "input")

comparison_dir = os.path.join(SAVE_DIR, "comparisons_best")
os.makedirs(comparison_dir, exist_ok=True)

# ========== LOAD DATA ==========
print("Loading dataset for metadata...")
dataset = load_dataset(DATASET_NAME, split="test")
clean_dataset = load_dataset(CLEAN_DATASET_NAME, split="test")

print("Loading best indices...")
best_base_indices = np.load("best_base_indices.npy")
best_control_indices = np.load("best_control_indices.npy")

def cosine_similarity(a, b):
    a = a.view(-1)
    b = b.view(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def embedding_to_text_image(embedding, width=256, height=20, font_size=12):
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    emb = embedding.detach().cpu().numpy().flatten()
    emb_text = " ".join(f"{x:.2f}" for x in emb[:10])  # Show first 10 dims

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((5, 2), emb_text, fill="black", font=font)
    return img

def create_comparison_grid(images, embeddings, labels, original_embedding, save_path):
    assert len(images) == len(labels) == len(embeddings)

    width, height = images[0].size
    font_size = 16
    label_height = 25
    text_height = 25
    similarity_height = 25
    total_height = height + label_height + text_height + similarity_height

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    result = Image.new("RGB", (width * len(images), total_height), "white")
    draw = ImageDraw.Draw(result)

    for i, (img, emb, label) in enumerate(zip(images, embeddings, labels)):
        x = i * width
        result.paste(img, (x, 0))

        text_width = draw.textlength(label, font=font)
        draw.text((x + (width - text_width) // 2, height + 2), label, fill="black", font=font)

        text_img = embedding_to_text_image(emb, width=width, height=text_height)
        result.paste(text_img, (x, height + label_height))

        if i > 0:
            sim = cosine_similarity(original_embedding, emb)
            sim_text = f"Cosine w/ original: {sim:.3f}"
            draw.text((x + 5, height + label_height + text_height + 2), sim_text, fill="black", font=font)

    result.save(save_path)

# ========== MAIN LOOP ==========
NUM_SAMPLES = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(NUM_SAMPLES):
    output_name = f"comparison_{i:03d}.png"
    comparison_path = os.path.join(comparison_dir, output_name)
    if os.path.exists(comparison_path):
        continue

    print(f"[{i+1}/{NUM_SAMPLES}] Processing case {i}...")

    sample = dataset[i]
    clean_sample = clean_dataset[i]

    try:
        # Load input + original
        input_img = Image.open(os.path.join(input_dir, f"{i:03d}.png")).convert("RGB")
        original_image = clean_sample["image"].resize((512, 512))

        input_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, sample["condition"]))).squeeze(0)
        original_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, clean_sample["condition"]))).squeeze(0)

        # Get best indices for this sample
        best_base_idx = int(best_base_indices[i])
        best_control_idx = int(best_control_indices[i])

        # Build paths
        base_img_path = os.path.join(base_dir, f"{i:03d}", f"{best_base_idx:03d}.png")
        controlnet_img_path = os.path.join(controlnet_dir, f"{i:03d}", f"{best_control_idx:03d}.png")

        base_embedding_path = os.path.join(SAVE_DIR, f"embeddings_base/{i:03d}/{best_base_idx:03d}.npy")
        controlnet_embedding_path = os.path.join(SAVE_DIR, f"embeddings_controlnet/{i:03d}/{best_control_idx:03d}.npy")

        # Load images and embeddings
        base_img = Image.open(base_img_path).convert("RGB")
        controlnet_img = Image.open(controlnet_img_path).convert("RGB")

        embedding_base = torch.from_numpy(np.load(base_embedding_path)).squeeze(0)
        embedding_controlnet = torch.from_numpy(np.load(controlnet_embedding_path)).squeeze(0)

        # Combine
        images = [original_image, input_img, base_img, controlnet_img]
        embeddings = [original_embedding, input_embedding, embedding_base, embedding_controlnet]
        labels = ["Original", "Masked", "Best Base", "Best ControlNet"]

        create_comparison_grid(images, embeddings, labels, original_embedding, comparison_path)

    except Exception as e:
        print(f"Failed to process case {i}: {e}")
        continue

print(f"\nAll comparisons saved in: {comparison_dir}")
