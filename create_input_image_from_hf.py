import os
import cv2
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

FFHQ_DATASET = "Ryan-sjtu/ffhq512-caption"
MASKS = "Milocas/celebahq_masked"
OUTPUT_DIR = "datasets/ffhq/masked_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading FFHQ and masks...")
ffhq = load_dataset(FFHQ_DATASET, split="train")
masks = load_dataset(MASKS, split="train")

limit = min(len(ffhq), len(masks))
print(f"Total aligned pairs available: {limit}")

for idx in tqdm(range(limit), desc="Creating FFHQ masked images"):
    try:
        img = ffhq[idx]["image"]
        img = np.array(img.convert("RGB"))  

        mask = masks[idx]["mask"]
        mask = np.array(mask.convert("L")) 

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.merge([mask, mask, mask])
        masked_img = cv2.bitwise_and(img, cv2.bitwise_not(mask_3ch))

        out_path = os.path.join(OUTPUT_DIR, f"{idx:05d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"Failed at index {idx}: {e}")
