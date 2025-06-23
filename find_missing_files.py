import os

folder_a = "datasets/celebahq/conditions_no_mask"
folder_b = "ControlNet_plus_plus/train/comparison_outputs/embeddings_base"

files_a = {os.path.splitext(f)[0] for f in os.listdir(folder_a) if f.endswith(".npy")}
files_b = {os.path.splitext(f)[0] for f in os.listdir(folder_b) if f.endswith(".npy")}

missing_in_b = files_a - files_b
missing_in_a = files_b - files_a

print(f"Total in A (expected): {len(files_a)}")
print(f"Total in B: {len(files_b)}\n")

if missing_in_a:
    print(f"Extra files in B (not in A): {len(missing_in_a)}")
    print(sorted(missing_in_a))
else:
    print("No extra files in B.")
