# test_unet_forward.py

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Create dummy inputs
batch_size = 4
height = width = 64
latent_channels = 4
hidden_states_dim = 77
cross_attention_dim = 1280

sample = torch.randn(batch_size, latent_channels, height, width)
timestep = torch.tensor([10] * batch_size)
encoder_hidden_states = torch.randn(batch_size, hidden_states_dim, cross_attention_dim)

# Instantiate the model (minimal config)
unet = UNet2DConditionModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    cross_attention_dim=1280,
    attention_head_dim=8,
)

# Forward pass
with torch.no_grad():
    out = unet(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )

print("Forward pass completed.")
