from datasets import load_from_disk

dataset_masked = load_from_disk("datasets/celebahq/hf_celebahq_masked")
dataset_clean = load_from_disk("datasets/celebahq/hf_celebahq_clean")

dataset_masked.push_to_hub("celebahq_masked")
dataset_clean.push_to_hub("celebahq_clean")