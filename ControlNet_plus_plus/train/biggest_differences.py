import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")  # For headless environments

# ========== CONFIG ==========
THRESHOLD = 0.05
DATASET_NAME = "Milocas/celebahq_masked"
CLEAN_DATASET_NAME = "Milocas/celebahq_clean"
SAVE_DIR = "./comparison_outputs_random_seed"
EMBEDDING_PREFIX = "../../"

base_dir = os.path.join(SAVE_DIR, "base")
controlnet_dir = os.path.join(SAVE_DIR, "controlnet")
input_dir = os.path.join(SAVE_DIR, "input")
comparison_dir = os.path.join(SAVE_DIR, "top_differences")
os.makedirs(comparison_dir, exist_ok=True)

# ========== LOAD DATASETS ==========
print("Loading datasets...")
dataset = load_dataset(DATASET_NAME, split="test")
clean_dataset = load_dataset(CLEAN_DATASET_NAME, split="test")

# ========== LOAD SIMILARITY DATA ==========
print("Loading cosine similarity results...")
npz = np.load("cosine_similarity_results_full.npz")
delta = npz["delta"]
indices = npz["indices"]

# ========== GROUPING ==========
print(f"Filtering samples with delta > {THRESHOLD} or < -{THRESHOLD}...")
improved_indices = indices[delta > THRESHOLD]
regressed_indices = indices[delta < -THRESHOLD]
print(f"Improved: {len(improved_indices)} | Regressed: {len(regressed_indices)}")

# ========== HELPERS ==========
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
    emb_text = " ".join(f"{x:.2f}" for x in emb[:10])
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((5, 2), emb_text, fill="black", font=font)
    return img

def draw_result_tag(image, tag_text, color):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (150, 25)], fill=color)
    draw.text((5, 5), tag_text, fill="white", font=font)
    return image

def create_comparison_grid(images, embeddings, labels, original_embedding, save_path, tag=None, color=None):
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

    if tag and color:
        draw_result_tag(result, tag, color)

    result.save(save_path)

# ========== MAIN LOOP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_group(group_name, indices_subset, tag_text, tag_color):
    for rank, idx in enumerate(indices_subset):
        idx = int(idx)
        name = f"sample_{idx:03d}.png"
        prefix = f"{group_name}_{rank:03d}_"
        save_path = os.path.join(comparison_dir, f"{prefix}{name}")
        print(f"[{group_name.upper()} {rank+1}/{len(indices_subset)}] Index {idx}")

        try:
            idx = int(idx)  
            sample = dataset[idx]
            clean_sample = clean_dataset[idx]

            input_img = Image.open(os.path.join(input_dir, name)).convert("RGB")
            base_img = Image.open(os.path.join(base_dir, name)).convert("RGB")
            controlnet_img = Image.open(os.path.join(controlnet_dir, name)).convert("RGB")

            input_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, sample["condition"]))).squeeze(0)
            original_embedding = torch.from_numpy(np.load(os.path.join(EMBEDDING_PREFIX, clean_sample["condition"]))).squeeze(0)

            base_embedding = torch.from_numpy(np.load(os.path.join(SAVE_DIR, f"embeddings_base/{idx:03d}.npy"))).squeeze(0)
            controlnet_embedding = torch.from_numpy(np.load(os.path.join(SAVE_DIR, f"embeddings_controlnet/{idx:03d}.npy"))).squeeze(0)

            original_image = clean_sample["image"].resize((512, 512))

            images = [original_image, input_img, base_img, controlnet_img]
            embeddings = [original_embedding, input_embedding, base_embedding, controlnet_embedding]
            labels = [
                "Original Image",
                "Masked Input",
                "Stable Diffusion Generation",
                "ControlNet Generation"
            ]

            create_comparison_grid(images, embeddings, labels, original_embedding, save_path, tag=tag_text, color=tag_color)

        except Exception as e:
            print(f"Failed to process index {idx}: {e}")
            continue

# Process improved and regressed groups
process_group("improved", improved_indices, "+Improved", "green")
process_group("regressed", regressed_indices, "-Regressed", "red")

print(f"\nAll comparison images saved in: {comparison_dir}")
