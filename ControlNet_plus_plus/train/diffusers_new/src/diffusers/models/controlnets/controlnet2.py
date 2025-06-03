# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_up_block,
)
from ..resnet import ResnetBlock2D, Upsample2D
from ..unets.unet_2d_condition import UNet2DConditionModel
from ..transformers.transformer_2d import Transformer2DModel
#from ..modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        output_states = ()
        
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                output_states += (hidden_states,)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                output_states += (hidden_states,)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                output_states += (hidden_states,)

        return hidden_states, output_states 


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        output_states = ()
        
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                output_states += (hidden_states,)
            else:
                hidden_states = resnet(hidden_states, temb)
                output_states += (hidden_states,)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                output_states += (hidden_states,)

        return hidden_states, output_states 
        
@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the middle block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """
    up_block_res_samples:Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


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
        #print(x.shape)
        x = self.block1(x, temb)
        #print(x.shape)
        x = self.upsample1(x)
        #print(x.shape)
        x = self.block2(x, temb)
        #print(x.shape)
        x = self.block3(x, temb)
        #print(x.shape)        
        x = self.out_conv(x)
        #print(x.shape)
       
        return x


class ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_dim: int = 512,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (1280, 1280, 640, 320),
        layers_per_block: int = 2,
        upsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        projection_class_embeddings_input_dim: Optional[int] = None,
        target_shape: Optional[Tuple[int, int, int]] = (1280, 8, 8),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        resnet_channel_config: Optional[List[List[Tuple[int, int]]]] = None,
        mid_block_channel: Optional[int] = None,
        cross_attention_norm: Optional[str] = None,
    ):
        super().__init__()

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim
        
        # Check inputs
        if len(block_out_channels) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `up_block_types`. `block_out_channels`: {block_out_channels}. `up_block_types`: {up_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `up_block_types`. `only_cross_attention`: {only_cross_attention}. `up_block_types`: {up_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `up_block_types`. `num_attention_heads`: {num_attention_heads}. `up_block_types`: {up_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(up_block_types)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        # self.conv_in = nn.Conv2d(
        #     in_channels, block_out_channels[0], kernel_size=3, padding=1
        #     )
        self.conv_in = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, block_out_channels[-1]//4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # 32x32 -> 16x16
            nn.Conv2d(block_out_channels[-1]//4, block_out_channels[-1]//2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # 16x16 -> 8x8
            nn.Conv2d(block_out_channels[-1]//2, block_out_channels[-1], kernel_size=3, stride=2, padding=1)
        )

        # time          
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )
        
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        
        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[-1],
            conditioning_dim=conditioning_dim,
            target_shape=target_shape
        )

        #print(self.controlnet_cond_embedding.parameters())
        
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(up_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(up_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(up_block_types)

        # for i in range(len(block_out_channels)):
        #     print(f"[DEBUG] block_out_channels[{i}]: {block_out_channels[i]}")

        self.up_blocks = nn.ModuleList([])
        self.controlnet_up_blocks = nn.ModuleList([])
        
        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        # print(f"midblock: {self.controlnet_mid_block}")
        
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        # up                
        self.up_blocks = nn.ModuleList()
        self.controlnet_up_blocks = nn.ModuleList([])
        self.num_upsamplers = 0  
        
        for i in range(len(block_out_channels)):
            print(f"[DEBUG] block_out_channels[{i}]: {block_out_channels[i]}") 

        for i in range(len(up_block_types)):
            print(f"[DEBUG] block_out_channels[{i}]: {up_block_types[i]}") 
        
        reversed_block_out_channels = list(reversed(block_out_channels))

        output_channel = reversed_block_out_channels[0]
        
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    dropout=0.0,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=transformer_layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    add_upsample=add_upsample,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            elif up_block_type == "UpBlock2D":
                up_block = UpBlock2D(
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    dropout=0.0,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_upsample=add_upsample,
                )
            else:
                raise ValueError(f"Unsupported up block type: {up_block_type}")
        
            self.up_blocks.append(up_block)
    
            print(f"[INIT] UpBlock {i}: {up_block_type} | in_channels={input_channel}, out_channels={output_channel}")

            if i == 0:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_up_blocks.append(controlnet_block)
            
            if not is_final_block:
                num_layers = layers_per_block + 1      
            else:
                num_layers = layers_per_block
                
            for _ in range(num_layers):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_up_blocks.append(controlnet_block)
                print(f"Added block (from layers_per_block): {controlnet_block}")

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_up_blocks.append(controlnet_block)
                print(f"Added block (extra): {controlnet_block}")
        
            print("\n")
        
        # Final residual block for the last UpBlock
        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_up_blocks.append(controlnet_block)
        
        print("\nFinal controlnet_up_blocks:")
        for i, block in enumerate(self.controlnet_up_blocks):
            print(f"[{i}] {block}")

        self.expected_down_block_res_shapes = [
            (320, 64, 64),
            (320, 64, 64),
            (320, 64, 64),
            (320, 32, 32),
            (640, 32, 32),
            (640, 32, 32),
            (640, 16, 16),
            (1280, 16, 16),
            (1280, 16, 16),
            (1280, 8, 8),
            (1280, 8, 8),
            (1280, 8, 8),
        ]


    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        target_shape = (
            unet.config.block_out_channels[-1],  # channels
            unet.config.sample_size // (2 ** (len(unet.config.block_out_channels) - 1)),  # height
            unet.config.sample_size // (2 ** (len(unet.config.block_out_channels) - 1))   # width
        )     
        
        conditioning_dim = unet.config.get("conditioning_dim", 512)
        mid_block_channel = unet.config.block_out_channels[-1]
        resnet_channel_config = []
        
        for up_block in unet.up_blocks:
            block_config = []
            for resnet in up_block.resnets:
                in_ch = resnet.conv1.in_channels
                out_ch = resnet.conv2.out_channels
                block_config.append((in_ch, out_ch))
            resnet_channel_config.append(block_config)

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            up_block_types=unet.config.up_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            upsample_padding=unet.config.get("upsample_padding", 1),
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift, 
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            conditioning_dim=conditioning_dim,
            target_shape=target_shape,
            resnet_channel_config=resnet_channel_config,
            mid_block_channel=mid_block_channel,
        )

        def compare_modules(control_mod, unet_mod, name):
            control_state = control_mod.state_dict()
            unet_state = unet_mod.state_dict()
            print(f"\n Comparing `{name}`")
            for k in control_state:
                if k in unet_state:
                    if control_state[k].shape == unet_state[k].shape:
                        print(f"  ✓ {k}: shape {control_state[k].shape}")
                    else:
                        print(f"  ⚠ {k}: shape mismatch - ControlNet: {control_state[k].shape}, UNet: {unet_state[k].shape}")
                else:
                    print(f"  ✘ {k}: not found in UNet")
        
        if load_weights_from_unet:
            #controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())
                
            for control_block, unet_block in zip(controlnet.up_blocks, unet.up_blocks):
                for i, (control_layer, unet_layer) in enumerate(zip(control_block.resnets, unet_block.resnets)):
                    control_state_dict = control_layer.state_dict()
                    unet_state_dict = unet_layer.state_dict()
                    for key in control_state_dict:
                        if key in unet_state_dict:
                            try:
                                if control_state_dict[key].shape != unet_state_dict[key].shape:
                                    if len(control_state_dict[key].shape) == len(unet_state_dict[key].shape):
                                        min_shape = tuple(min(s1, s2) for s1, s2 in zip(control_state_dict[key].shape, unet_state_dict[key].shape))
                                        slices = tuple(slice(0, s) for s in min_shape)
                                        resized_weight = torch.zeros_like(control_state_dict[key])
                                        resized_weight[slices] = unet_state_dict[key][slices]
                                        control_state_dict[key] = resized_weight
                                    else:
                                        print(f"Shape mismatch for {key}, reinitializing.")
                                        nn.init.xavier_uniform_(control_state_dict[key])
                                else:
                                    control_state_dict[key] = unet_state_dict[key]
                            except Exception as e:
                                print(f"Error resizing {key}: {e}")
                    control_layer.load_state_dict(control_state_dict, strict=False)
            
                # Handle upsamplers
                if control_block.upsamplers and unet_block.upsamplers:
                    for j, (control_upsample, unet_upsample) in enumerate(zip(control_block.upsamplers, unet_block.upsamplers)):
                        try:
                            control_upsample.load_state_dict(unet_upsample.state_dict(), strict=False)
                        except RuntimeError as e:
                            print(f"Error in upsampler {j}: {e}")

            #controlnet.up_blocks.load_state_dict(unet.up_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        # if hasattr(controlnet, "up_blocks") and hasattr(unet, "up_blocks"):
        #         for i, (c_blk, u_blk) in enumerate(zip(controlnet.up_blocks, unet.up_blocks)):
        #             compare_modules(c_blk, u_blk, f"up_blocks[{i}]")
        
        return controlnet

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.FloatTensor] = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnets.controlnet.ControlNetOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.controlnets.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnets.controlnet.ControlNetOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        # # check channel order
        # channel_order = self.config.controlnet_conditioning_channel_order

        # if channel_order == "rgb":
        #     # in rgb order by default
        #     ...
        # elif channel_order == "bgr":
        #     controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        # else:
        #     raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
            
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        #print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process        
        # print(f"Input sample shape: {sample.shape}")
        sample = self.conv_in(sample)
        # print(f"After conv_in shape: {sample.shape}")

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # print(f"ControlNet condition shape: {controlnet_cond.shape}")

        sample = sample + controlnet_cond
        # print(f"After all shape: {sample.shape}")
        
        # 3. Mid block
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention"):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # print(f"sample chape before midblock:{sample.shape}")
        mid_block_res_sample = self.controlnet_mid_block(sample)
        # print(f"Mid block output shape: {mid_block_res_sample.shape}")

        # 4. up
        up_block_res_samples = (sample,)

        down_block_res_samples = [
            torch.zeros(sample.shape[0], c, h, w, dtype=sample.dtype, device=sample.device)
            for (c, h, w) in self.expected_down_block_res_shapes
        ]
        # print("\n")
        # for i, tensor in enumerate(down_block_res_samples):
        #     print(f"[DEBUG] down_block_res_samples[{i}]: shape = {tensor.shape}")
        
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            in_res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
                
            # print(f"\n[DEBUG] Block {i} — feeding {len(in_res_samples)} skip tensors:")
            # for j, res in enumerate(in_res_samples):
            #     print(f"  current_res[{j}] shape: {res.shape}")
            
            # print(f"  - hidden_states (input to upsample_block): {sample.shape}")
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=in_res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=None,
                )
            else:
                sample, res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=in_res_samples,
                    upsample_size=upsample_size,
                )

    
            print(f"[DEBUG] upsample_block: {upsample_block.__class__.__name__}")
            print(f"[DEBUG] res_samples is a {type(res_samples)} of length {len(res_samples)}")
            for i, res in enumerate(res_samples):
                print(f"    res_samples[{i}] shape: {res.shape}")
                
            up_block_res_samples += tuple(res_samples)

        print("")
        for i, tensor in enumerate(up_block_res_samples):
            print(f"[DEBUG] up_block_res_samples[{i}]: shape = {tensor.shape}")

        controlnet_up_block_res_samples = ()

        for up_block_res_sample, controlnet_block in zip(up_block_res_samples, self.controlnet_up_blocks):
            up_block_res_sample = controlnet_block(up_block_res_sample)
            controlnet_up_block_res_samples = controlnet_up_block_res_samples + (up_block_res_sample,)
        
        up_block_res_samples = controlnet_up_block_res_samples 

        
        # 5. Scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(up_block_res_samples) + 1, device=sample.device)
            scales = scales * conditioning_scale
            up_block_res_samples = [sample * scale for sample, scale in zip(up_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            up_block_res_samples = [sample * conditioning_scale for sample in up_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale
    
        if self.config.global_pool_conditions:
            up_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in up_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)
        
        if not return_dict:
            return (mid_block_res_sample, up_block_res_samples)

        return ControlNetOutput(
            mid_block_res_sample=mid_block_res_sample,
            up_block_res_samples=up_block_res_samples            
        )


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module