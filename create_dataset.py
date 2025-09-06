import os 
from datasets import Dataset, Features, Value, Image, DatasetDict
from tqdm import tqdm

def to_columns(records):
    columns = {"image": [], "condition": [], "mask": []}
    for r in records:
        columns["image"].append(r["image"])
        columns["condition"].append(r["condition"])
        columns["mask"].append(r["mask"])
    return columns

def build_celebahq_dataset(image_dir, condition_dir, mask_file, output_path):
    image_files = sorted(os.listdir(image_dir))
    
    records = []
    for fname in tqdm(image_files, desc=f"Building dataset for {image_dir}"):
        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        condition_path = os.path.join(condition_dir, f"{base_name}.npy")
        # mask_path = os.path.join(mask_dir, f"{base_name}.png")
        if not os.path.exists(image_path):
            print("Missing image:", image_path)
        if not os.path.exists(condition_path):
            print("Missing condition:", condition_path)
        if not os.path.exists(mask_file):
            print("Missing mask file:", mask_file)    
        
        if os.path.exists(image_path) and os.path.exists(condition_path) and os.path.exists(mask_file):
            records.append({
                "image": image_path,
                "condition": condition_path,
                "mask": mask_file
            })


    features = Features({
        "image": Image(),
        "condition": Value("string"),  # path to .npy, load later
        "mask": Image()
    })

    total = len(records)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    print(f"Total records: {total} (Train: {train_size}, Val: {val_size}, Test: {test_size})")

    train_records = to_columns(records[:train_size])
    val_records = to_columns(records[train_size:train_size + val_size])
    test_records = to_columns(records[train_size + val_size:])
    
    ds_train = Dataset.from_dict(train_records, features=features)
    ds_val   = Dataset.from_dict(val_records, features=features)
    ds_test  = Dataset.from_dict(test_records, features=features)

    dataset = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })
    dataset.save_to_disk(output_path)
    print(f"Saved {len(dataset)} samples to: {output_path}")

base_path = "datasets/celebahq"

# Dataset 1: With occlusion
build_celebahq_dataset(
    image_dir=os.path.join(base_path, "image"),
    condition_dir=os.path.join(base_path, "conditions_no_mask"),
    mask_file=os.path.join(base_path, "mask.png"),
    output_path=os.path.join(base_path, "hf_celebahq_mask")
)

# # Dataset 2: Without occlusion
# build_celebahq_dataset(
#     image_dir=os.path.join(base_path, "image"),
#     condition_dir=os.path.join(base_path, "conditions_no_mask"),
#     mask_dir=os.path.join(base_path, "mask"),
#     output_path=os.path.join(base_path, "hf_celebahq_clean")
# )
