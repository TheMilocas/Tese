from diffusers import StableDiffusionPipeline

import torch
import torch.nn as nn

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# unet = pipe.unet


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

# block = unet.down_blocks[0]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

# block = unet.down_blocks[1]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

# block = unet.down_blocks[2]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

# block = unet.up_blocks[1]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

# block = unet.up_blocks[2]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")

# block = unet.up_blocks[3]

# print("Submodules in CrossAttnUpBlock2D:")
# for name, module in block.named_modules():
#     if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#         print(f"{name}: cross_attention_dim = {module.config.cross_attention_dim}")


# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

# for i, block in enumerate(controlnet.down_blocks):
#     print(f"\nControlNet Down Block {i}")
#     for name, module in block.named_modules():
#         if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#             print(f"  {name}: cross_attention_dim = {module.config.cross_attention_dim}")

# for i, block in enumerate(controlnet.up_blocks):
#     print(f"\nControlNet Up Block {i}")
#     for name, module in block.named_modules():
#         if hasattr(module, "config") and hasattr(module.config, "cross_attention_dim"):
#             print(f"  {name}: cross_attention_dim = {module.config.cross_attention_dim}")

import torch
from diffusers import UNet2DConditionModel
import types

# Load UNet from diffusers
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# Create dummy inputs
sample = torch.randn(1, 4, 64, 64)
timestep = torch.tensor([1], dtype=torch.long)
encoder_hidden_states = torch.randn(1, 77, 768)

# Patch each down block
for i, block in enumerate(unet.down_blocks):
    orig_forward = block.forward
    def make_forward(i, orig_forward):
        def wrapped(self, *args, **kwargs):
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                print(f" Down Block {i} residuals:")
                for j, res in enumerate(out[1]):
                    print(f"  - res[{j}] shape: {res.shape}")
            return out
        return types.MethodType(wrapped, block)
    block.forward = make_forward(i, orig_forward)

# Patch mid block
if hasattr(unet, "mid_block") and unet.mid_block is not None:
    orig_mid = unet.mid_block.forward
    def mid_forward(self, *args, **kwargs):
        out = orig_mid(*args, **kwargs)
        print(f" Mid Block output shape: {out.shape}")
        return out
    unet.mid_block.forward = types.MethodType(mid_forward, unet.mid_block)

# Patch each up block
for i, block in enumerate(unet.up_blocks):
    orig_forward = block.forward
    def make_forward(i, orig_forward):
        def wrapped(self, *args, **kwargs):
            out = orig_forward(*args, **kwargs)
            print(f" Up Block {i} output shape: {out.shape}")
            return out
        return types.MethodType(wrapped, block)
    block.forward = make_forward(i, orig_forward)

# Run forward
out = unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)


