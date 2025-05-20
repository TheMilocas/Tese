from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn

# Load the model (only UNet part is needed)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipe.unet

# Function to recursively print Conv2d layers with shape info
def print_conv_layers(module, prefix=""):
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.Conv2d):
            print(f"{full_name}: {layer}")
        else:
            print_conv_layers(layer, full_name)

# Print all Conv2d layers inside the UNet
print_conv_layers(unet)
