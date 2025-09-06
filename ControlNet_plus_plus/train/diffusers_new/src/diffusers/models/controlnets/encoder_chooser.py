import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from torchinfo import summary 
from torch.nn import functional as F
from ControlNet_plus_plus.train.diffusers_new.src.diffusers.models.resnet import ResnetBlock2D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self, 
        conditioning_embedding_channels: int, 
        conditioning_dim: int = 512,
        target_shape: Tuple[int, int, int] = (1280, 8, 8)
        ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(conditioning_dim, 128 * 4 * 4),
            nn.SiLU()
        )
        self.block1 = ResnetBlock2D(in_channels=128, out_channels=256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # 4x4 → 8x8

        self.block2 = ResnetBlock2D(in_channels=256, out_channels=512)
        self.block3 = ResnetBlock2D(in_channels=512, out_channels=target_shape[0])

        self.out_conv = zero_module(
            nn.Conv2d(target_shape[0], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
             
        x = self.fc(x).view(x.size(0), 128, 4, 4)  # B, 128, 4, 4

        temb_channels = getattr(self.block1, "temb_channels", 512)
        temb = torch.zeros(x.size(0), temb_channels, device=x.device)
        x = self.block1(x, temb)
        x = self.upsample1(x)
        x = self.block2(x, temb)
        x = self.block3(x, temb)
        x = self.out_conv(x)
       
        return x


class OriginalControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int = 1280,
        input_channels: int = 512,
        block_out_channels: tuple = (512, 640, 1280),
    ):
        super().__init__()

        self.blocks = nn.ModuleList([])
        in_ch = input_channels

        for out_ch in block_out_channels:
            self.blocks.append(nn.Conv2d(in_ch, in_ch, kernel_size=1))
            self.blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            in_ch = out_ch

        self.blocks.append(nn.Conv2d(in_ch, in_ch, kernel_size=1))
        
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:  # [B, 512]
            x = x.unsqueeze(-1).unsqueeze(-1)  
    
        for block in self.blocks:
            x = F.silu(block(x))
        x = self.conv_out(x)
        return x


class AFCE(nn.Module):
    def __init__(
        self, 
        conditioning_embedding_channels=1280, 
        conditioning_dim=512,
        target_shape=(1280, 8, 8) 
    ):
        super().__init__()
        
        self.target_shape = target_shape 
        
        self.fc = nn.Linear(conditioning_dim, 512 * 8 * 8)
        self.proj = nn.Conv2d(512, conditioning_embedding_channels, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            ResnetBlock2D(in_channels=conditioning_embedding_channels, out_channels=conditioning_embedding_channels),
            ResnetBlock2D(in_channels=conditioning_embedding_channels, out_channels=conditioning_embedding_channels),
        )

        self.out_conv = zero_module(
            nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, 1)
        )

    def forward(self, x, temb=None):
        b = x.size(0)
        x = self.fc(x).view(b, 512, *self.target_shape[1:])  
        x = self.proj(x)  
        if temb is None:
            temb = torch.zeros(
                b, getattr(self.resblocks[0], "temb_channels", 512), device=x.device
            )
        for block in self.resblocks:
            x = block(x, temb)
        x = self.out_conv(x)
        return x

class IdentityUpsampleEncoder(nn.Module):
    def __init__(
        self, 
        conditioning_embedding_channels=1280, 
        conditioning_dim=512, 
        target_shape=(1280, 8, 8)
    ):
        super().__init__()
        c_out, H, W = target_shape
        self.fc = nn.Linear(conditioning_dim, 512 * 4 * 4)
        self.expand_conv = nn.Conv2d(512, c_out, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  
        self.out_conv = zero_module(nn.Conv2d(c_out, conditioning_embedding_channels, kernel_size=3, padding=1))

    def forward(self, x):
        B = x.size(0)
        x = self.fc(x).view(B, 512, 4, 4)   # B, 512, 4, 4
        x = self.upsample(x)                # B, 512, 8, 8
        x = self.expand_conv(x)             # B, 1280, 8, 8
        x = self.out_conv(x)                # B, 1280, 8, 8
        return x
    

if __name__ == "__main__":
    img_model = OriginalControlNetConditioningEmbedding(
        conditioning_embedding_channels=320,
        conditioning_channels=3
    )
    vec_model = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=1280,
        conditioning_dim=512,
        target_shape=(1280, 8, 8)
    )
    alt_model = ConditioningEmbedding(
        conditioning_embedding_channels=1280,
        input_channels=512,
        block_out_channels=(512, 640, 1280)
    )
    alt_model1 = AFCE(
        conditioning_embedding_channels=1280,
        conditioning_dim=512,
        target_shape=(1280, 8, 8)
    )
    alt_model2 = IdentityUpsampleEncoder(
        conditioning_embedding_channels=1280,
        conditioning_dim=512,
        target_shape=(1280, 8, 8)
    )
    
    print("Image-based ControlNet encoder params:", count_parameters(img_model))
    print("encoder params:", count_parameters(vec_model))
    print("similar net encoder params:", count_parameters(alt_model))
    print("AFCE encoder params:", count_parameters(alt_model1))
    print("new net encoder params:", count_parameters(alt_model2))

    # Optional: print layer summary
    dummy_img = torch.randn(1, 3, 512, 512)
    dummy_vec = torch.randn(1, 512)
    # summary(img_model, input_data=dummy_img)
    # summary(vec_model, input_data=dummy_vec)
    summary(alt_model, input_data=dummy_vec)
    # summary(alt_model1, input_data=dummy_vec)
    # summary(alt_model2, input_data=dummy_vec)
