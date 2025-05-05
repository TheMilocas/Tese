import os
import json
from tqdm import tqdm

image_dir = 'datasets/celebahq/masked_images'
condition_dir = 'datasets/celebahq/conditions'
output_json_path = 'datasets/celebahq/metadata.json'

metadata = []

image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for img_name in tqdm(image_filenames):
    img_path = os.path.join('masked_images', img_name)
    condition_filename = os.path.splitext(img_name)[0] + '.npy'
    condition_path = os.path.join('conditions', condition_filename)

    entry = {
        "image": img_path,
        "condition": condition_path,
        "prompt": "" 
    }

    metadata.append(entry)

with open(output_json_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata JSON saved to {output_json_path} with {len(metadata)} entries.")
