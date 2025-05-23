from diffusers import StableDiffusionPipeline

import torch
import torch.nn as nn

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipe.unet


def print_conv_layers(module, prefix=""):
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.Conv2d):
            print(f"{full_name}: {layer}")
        else:
            print_conv_layers(layer, full_name)


# print_conv_layers(unet)

# Print types of down blocks
# print("Down Blocks:")
# for i, block in enumerate(unet.down_blocks):
#     print(f"  Block {i}: {block.__class__.__name__}")

# # Print types of up blocks
# print("\nUp Blocks:")
# for i, block in enumerate(unet.up_blocks):
#     print(f"  Block {i}: {block.__class__.__name__}")

# # Print type of mid block
# print("\nMid Block:")
# print(f"  {unet.mid_block.__class__.__name__}")

# Let's inspect the submodules inside up_block 1 (first CrossAttnUpBlock2D)

block = unet.down_blocks[0]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

block = unet.down_blocks[1]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

block = unet.down_blocks[2]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

block = unet.up_blocks[1]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

block = unet.up_blocks[2]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

block = unet.up_blocks[3]

print("Submodules in CrossAttnUpBlock2D:")
for name, module in block.named_modules():
    if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
        print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

for i, block in enumerate(controlnet.down_blocks):
    print(f"\nControlNet Down Block {i}")
    for name, module in block.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
            print(f"  {name}: cross_attention_dim = {module.config.cross_attention_dim}")

# for i, block in enumerate(controlnet.up_blocks):
#     print(f"\nControlNet Up Block {i}")
#     for name, module in block.named_modules():
#         if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#             print(f"  {name}: cross_attention_dim = {module.config.cross_attention_dim}")