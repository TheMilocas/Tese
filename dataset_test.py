from datasets import Dataset, Features, Image, Value, Array2D
import json
import numpy as np
from PIL import Image as PILImage
import os

# Load the JSON metadata
with open("idi-10.json") as f:
    meta = json.load(f)

# Define features (adjust based on your needs)
features = Features({
    "image": Image(),
    "identity_embedding": Array2D(shape=(512,), dtype="float32"),  # ARCFACE dim
    "image_id": Value("string"),
    "split": Value("string"),  # "train", "val", "test"
})

def build_dataset(root_dir, meta):
    data = []
    for split in ["train", "val", "test"]:
        image_dir = os.path.join(root_dir, f"{split}_images/all")
        for img_file in os.listdir(image_dir):
            if img_file.endswith(".jpg") or img_file.endswith(".png"):
                img_id = img_file.split(".")[0]
                embedding = np.load(f"identity_embeddings/{img_id}.npy")  
                data.append({
                    "image": os.path.join(image_dir, img_file),
                    "identity_embedding": embedding,
                    "image_id": img_id,
                    "split": split,
                })
    return Dataset.from_list(data, features=features)

dataset = build_dataset("path/to/root", meta)

print(dataset[0])