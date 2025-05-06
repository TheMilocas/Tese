import os
import json
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

dataset_root = "datasets/celebahq"
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}  # Adjust as needed

with open(f"{dataset_root}/metadata.json") as f:
    metadata = json.load(f)

all_files = set(os.path.splitext(f)[0] for f in os.listdir(f"{dataset_root}/image"))
basenames = list(all_files)

train_val, test = train_test_split(basenames, test_size=split_ratios['test'], random_state=42)
train, val = train_test_split(train_val, test_size=split_ratios['val']/(1-split_ratios['test']), random_state=42)

for split, names in [('train', train), ('val', val), ('test', test)]:
    os.makedirs(f"{dataset_root}/{split}/images", exist_ok=True)
    os.makedirs(f"{dataset_root}/{split}/masked", exist_ok=True)
    os.makedirs(f"{dataset_root}/{split}/masks", exist_ok=True)
    os.makedirs(f"{dataset_root}/{split}/emb_masked", exist_ok=True)
    os.makedirs(f"{dataset_root}/{split}/emb_clean", exist_ok=True)
    
    for name in names:
        for src, dst in [
            (f"image/{name}.jpg", f"{split}/images"),
            (f"masked_images/{name}.jpg", f"{split}/masked"),
            (f"mask/{name}.png", f"{split}/masks"),
            (f"conditions/{name}.npy", f"{split}/emb_masked"),
            (f"conditions_no_mask/{name}.npy", f"{split}/emb_clean")
        ]:
            src_path = f"{dataset_root}/{src}"
            dst_path = f"{dataset_root}/{dst}"
            if os.path.exists(src_path):
                os.symlink(os.path.abspath(src_path), f"{dst_path}/{os.path.basename(src)}")