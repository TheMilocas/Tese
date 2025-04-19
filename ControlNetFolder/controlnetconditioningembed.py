import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import Tuple

def zero_module(module):
    # Helper function from original ControlNet
    for p in module.parameters():
        p.detach().zero_()
    return module
    
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_dim: int = 512,  # ArcFace vector dimension
        target_shape: Tuple[int, int, int] = (1280, 8, 8),  # Default middle block shape (channels, height, width)
    ):
        super().__init__()
        self.target_shape = target_shape
        channels, height, width = target_shape
        
        self.fc = nn.Linear(conditioning_dim, channels * height * width)
        
        self.conv_out = zero_module(
            nn.Conv2d(channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        print(f"Input conditioning shape: {conditioning.shape}")  # (batch_size, 512)
        
        batch_size = conditioning.shape[0]
        embedding = self.fc(conditioning)
        print(f"After FC layer shape: {embedding.shape}")  # (batch_size, channels*height*width)
        
        embedding = embedding.view(batch_size, *self.target_shape)
        print(f"After reshape shape: {embedding.shape}")  # target_shape
        
        embedding = self.conv_out(embedding)
        print(f"Final output shape: {embedding.shape}")  # final shape after zero conv
        
        return embedding

class ControlNetConditioningEmbedding1(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_dim: int = 512,  # ArcFace vector dimension
        target_shape: Tuple[int, int, int] = (1280, 8, 8),  # Default middle block shape (channels, height, width)
    ):
        super().__init__()
        
        self.target_shape = target_shape
        channels, height, width = target_shape
        self.conditioning_dim = conditioning_dim
        
        self.fc = nn.Linear(conditioning_dim, conditioning_dim * 1 * 1)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # 1x1 -> 2x2
            nn.Conv2d(conditioning_dim, conditioning_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # 2x2 -> 4x4
            nn.Conv2d(conditioning_dim, conditioning_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # 4x4 -> 8x8
            nn.Conv2d(conditioning_dim, target_shape[0], kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, conditioning):
        print(f"Input conditioning shape: {conditioning.shape}")     # (batch_size, 512)
        
        embedding = self.fc(conditioning)                            
        print(f"After FC layer shape: {embedding.shape}")            # (batch, 512*1*1)
        
        embedding = embedding.view(-1, self.conditioning_dim, 1, 1)  
        print(f"After reshape shape: {embedding.shape}")             # (batch, 512, 1, 1)
        
        embedding = self.up1(embedding)                              
        print(f"After Upsample layer shape: {embedding.shape}")      # → (512, 2, 2)
        
        embedding = self.up2(embedding)                              
        print(f"After Upsample layer shape: {embedding.shape}")      # → (512, 4, 4)
        
        embedding = self.up3(embedding)                              
        print(f"Final output shape: {embedding.shape}")              # → (1280, 8, 8)
        
        return embedding


batch_size = 2
conditioning_dim = 512        # ArcFace embedding size 
target_shape = (1280, 8, 8)   # Middle block shape
embedding_channels = 1280     # UNet's middle block channels

embedding_net = ControlNetConditioningEmbedding(
    conditioning_embedding_channels=embedding_channels,
    conditioning_dim=conditioning_dim,
    target_shape=target_shape
)

dummy_input = torch.randn(batch_size, conditioning_dim)
print(f"\nCreated dummy input: {dummy_input.shape}")

# Run test
print("\nRunning forward pass...")
output = embedding_net(dummy_input)

summary(embedding_net, (512,))

print("\n")
print("Final output shape:", output.shape)