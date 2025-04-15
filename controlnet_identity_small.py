from diffusers import UNet2DConditionModel
import torch.nn as nn
import torch
from diffusers.models.controlnet import zero_conv

# This converts 512D identity embedding to mid-block input (16x16x768 for example)
class IdentityEmbeddingToFeatureMap(nn.Module):
    def __init__(self, embedding_dim=512, out_channels=768, spatial_size=16):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, out_channels * spatial_size * spatial_size),
            nn.ReLU()
        )
        
    def forward(self, identity_embedding):
        batch_size = identity_embedding.shape[0]
        x = self.fc(identity_embedding)
        x = x.view(batch_size, self.out_channels, self.spatial_size, self.spatial_size)
        return x

class IdentityControlNet(nn.Module):
    def __init__(self, unet_config, identity_embed_dim=512):
        super().__init__()
        self.unet = UNet2DConditionModel(**unet_config)
        self.injector = IdentityEmbeddingToFeatureMap(
            embedding_dim=identity_embed_dim,
            out_channels=self.unet.config.block_out_channels[-1],
            spatial_size=16  # Adjust depending on the SD model resolution
        )

    def forward(
        self,
        sample,  # latent input
        timestep,
        encoder_hidden_states,
        identity_embedding=None,  # [batch, 512]
        return_dict=True,
        **kwargs,
    ):
        # Get original mid-block input
        mid_input = self.injector(identity_embedding)

        # Let’s pass it into the U-Net and override mid_block
        # NOTE: This is an example and may need to adjust for your UNet config

        # You will need to inject `mid_input` into self.unet.forward() logic
        # If you're overriding just mid-block: use hooks or modify forward() manually

        # For a quick prototype:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            return_dict=return_dict,
        )
