import os
import torch
import timm
from huggingface_hub import hf_hub_download

# Correct call: Download the model file directly from the iresnet100 folder
checkpoint_path = hf_hub_download(
    repo_id="deepinsight/arcface_torch",
    filename="iresnet100/model.pth"  
)

# Load the architecture — 'resnet100' is used by iresnet100
model = timm.create_model('resnet100', num_classes=512, pretrained=False)

# Load the weights
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)

# Save locally
save_path = 'ARCFACE/models/iresnet100.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)

print(f"Saved iresnet100 model to: {save_path}")
