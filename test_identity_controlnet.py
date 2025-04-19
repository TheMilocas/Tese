import torch
from torch import nn
from diffusers.models.controlnets.controlnet import (
    ControlNetModel,
    ControlNetOutput,
    ControlNetConditioningEmbedding,
    zero_module
)
from diffusers.models.attention_processor import (
    AttentionProcessor,
    AttnProcessor,
    AttnAddedKVProcessor
)

def get_test_inputs(batch_size=2):
    return {
        "sample": torch.randn(batch_size, 4, 64, 64),
        "timestep": 1,
        "encoder_hidden_states": torch.randn(batch_size, 77, 768),
        "controlnet_cond": torch.randn(batch_size, 512),  # ArcFace vectors
        "conditioning_scale": 1.0,
        "return_dict": True
    }

def test_controlnet_model_initialization():
    print("Running: test_controlnet_model_initialization")
    model = ControlNetModel(
        conditioning_dim=512,
        target_shape=(1280, 8, 8),
        block_out_channels=(320, 640, 1280, 1280)
    )
    
    assert hasattr(model, 'controlnet_cond_embedding')
    assert model.controlnet_cond_embedding.fc.in_features == 512
    assert model.controlnet_cond_embedding.fc.out_features == 1280 * 8 * 8
    print("Passed")

def test_forward_pass_shapes():
    print("Running: test_forward_pass_shapes")
    model = ControlNetModel(
        conditioning_dim=512,
        target_shape=(1280, 8, 8),
        block_out_channels=(320, 640, 1280, 1280)
    ).eval()
    
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64)
    timestep = 1
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    controlnet_cond = torch.randn(batch_size, 512)
    
    with torch.no_grad():
        output = model(
            sample,
            timestep,
            encoder_hidden_states,
            controlnet_cond,
            return_dict=True
        )
    
    assert isinstance(output, ControlNetOutput)
    assert len(output.down_block_res_samples) == len(model.up_blocks)
    assert output.mid_block_res_sample.shape == (batch_size, 1280, 8, 8)

    for i, res_sample in enumerate(output.down_block_res_samples):
        expected_channels = model.block_out_channels[::-1][i]
        assert res_sample.shape[1] == expected_channels
    print("Passed")

def test_from_unet_loading():
    print("Running: test_from_unet_loading")
    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
            self.mid_block = nn.Conv2d(1280, 1280, kernel_size=3, padding=1)
            self.up_blocks = nn.ModuleList([nn.Conv2d(1280, 640, kernel_size=3, padding=1)])
            self.config = lambda: None
            self.config.in_channels = 4
            self.config.block_out_channels = (320, 640, 1280, 1280)

    dummy_unet = DummyUNet()
    controlnet = ControlNetModel.from_unet(
        dummy_unet,
        load_weights_from_unet=True,
        conditioning_channels=512
    )

    assert torch.allclose(controlnet.conv_in.weight, dummy_unet.conv_in.weight)
    assert torch.allclose(controlnet.mid_block.weight, dummy_unet.mid_block.weight)
    print("Passed")

def test_controlnet_output_order():
    print("Running: test_controlnet_output_order")
    model = ControlNetModel(
        conditioning_dim=512,
        block_out_channels=(320, 640, 1280)
    ).eval()

    dummy_input = torch.randn(1, 4, 64, 64)
    output = model(
        dummy_input,
        timestep=1,
        encoder_hidden_states=None,
        controlnet_cond=torch.randn(1, 512)
    )

    resolutions = [sample.shape[-1] for sample in output.down_block_res_samples]
    assert resolutions == sorted(resolutions, reverse=True), "Outputs should be high-res to low-res"
    print("Passed")

def test_gradient_checkpointing():
    print("Running: test_gradient_checkpointing")
    model = ControlNetModel(
        conditioning_dim=512,
        gradient_checkpointing=True
    )
    assert model._supports_gradient_checkpointing
    assert any(hasattr(m, 'gradient_checkpointing') for m in model.modules())
    print("Passed")

def test_attention_processors():
    print("Running: test_attention_processors")
    model = ControlNetModel(conditioning_dim=512)

    processors = model.attn_processors
    assert len(processors) > 0

    new_processor = AttnProcessor()
    model.set_attn_processor(new_processor)
    assert all(isinstance(p, AttnProcessor) for p in model.attn_processors.values())
    print("Passed")

if __name__ == "__main__":
    test_controlnet_model_initialization()
    test_forward_pass_shapes()
    test_from_unet_loading()
    test_controlnet_output_order()
    test_gradient_checkpointing()
    test_attention_processors()
