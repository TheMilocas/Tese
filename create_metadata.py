import os
import json

dataset_root = "datasets/celebahq"
splits = ['train', 'val', 'test']

for split in splits:
    metadata = []
    image_dir = f"{dataset_root}/{split}/images"
    
    for img_file in os.listdir(image_dir):
        base = os.path.splitext(img_file)[0]
        
        entry = {
            "image": f"{split}/images/{img_file}",
            "masked_image": f"{split}/masked/{base}.jpg",
            "mask": f"{split}/masks/{base}.png",
            "embedding_masked": f"{split}/emb_masked/{base}.npy",
            "embedding_clean": f"{split}/emb_clean/{base}.npy",
            "prompt": "" 
        }

        if all(os.path.exists(f"{dataset_root}/{v}") for k,v in entry.items() if k != 'prompt'):
            metadata.append(entry)
    
    with open(f"{dataset_root}/{split}_metadata.jsonl", 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')