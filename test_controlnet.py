import torch
import pytest
from torch import nn
from your_module import ControlNetModel, ControlNetOutput  # Replace with actual import

def test_controlnet_model_initialization():
    """Test that the model initializes with vector conditioning"""
    model = ControlNetModel(
        conditioning_dim=512,
        target_shape=(1280, 8, 8),
        block_out_channels=(320, 640, 1280, 1280)
    )
    
    assert hasattr(model, 'controlnet_cond_embedding')
    assert model.controlnet_cond_embedding.fc.in_features == 512
    assert model.controlnet_cond_embedding.fc.out_features == 1280 * 8 * 8

def test_forward_pass_shapes():
    """Verify tensor shapes throughout the forward pass"""
    model = ControlNetModel(
        conditioning_dim=512,
        target_shape=(1280, 8, 8),
        block_out_channels=(320, 640, 1280, 1280)
    ).eval()
    
    # Test inputs
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64)  # Latents
    timestep = 1
    encoder_hidden_states = torch.randn(batch_size, 77, 768)  # Text embeddings
    controlnet_cond = torch.randn(batch_size, 512)  # ArcFace vectors
    
    # Run forward pass
    with torch.no_grad():
        output = model(
            sample,
            timestep,
            encoder_hidden_states,
            controlnet_cond,
            return_dict=True
        )
    
    # Verify output structure
    assert isinstance(output, ControlNetOutput)
    assert len(output.down_block_res_samples) == len(model.up_blocks)
    assert output.mid_block_res_sample.shape == (batch_size, 1280, 8, 8)
    
    # Verify each decoder block output
    for i, res_sample in enumerate(output.down_block_res_samples):
        expected_channels = model.block_out_channels[::-1][i]
        assert res_sample.shape[1] == expected_channels

def test_from_unet_loading():
    """Test weight loading from UNet"""
    # Create dummy UNet (in practice you'd load a real one)
    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
            self.mid_block = nn.Conv2d(1280, 1280, kernel_size=3, padding=1)
            self.up_blocks = nn.ModuleList([nn.Conv2d(1280, 640, kernel_size=3, padding=1)])
            self.config = lambda: None  # Mock config
            self.config.in_channels = 4
            self.config.block_out_channels = (320, 640, 1280, 1280)
            # Add other required config attributes...
    
    dummy_unet = DummyUNet()
    controlnet = ControlNetModel.from_unet(
        dummy_unet,
        load_weights_from_unet=True,
        conditioning_channels=512  # ArcFace dim
    )
    
    # Verify weights were copied
    assert torch.allclose(
        controlnet.conv_in.weight,
        dummy_unet.conv_in.weight
    )
    assert torch.allclose(
        controlnet.mid_block.weight,
        dummy_unet.mid_block.weight
    )

def test_controlnet_output_order():
    """Verify decoder outputs are properly reversed"""
    model = ControlNetModel(
        conditioning_dim=512,
        block_out_channels=(320, 640, 1280)
    ).eval()
    
    # Run dummy forward pass
    dummy_input = torch.randn(1, 4, 64, 64)
    output = model(
        dummy_input,
        timestep=1,
        encoder_hidden_states=None,
        controlnet_cond=torch.randn(1, 512)
    )
    
    # Verify order is high-res to low-res
    resolutions = [sample.shape[-1] for sample in output.down_block_res_samples]
    assert resolutions == sorted(resolutions, reverse=True), "Outputs should be high-res to low-res"

def test_gradient_checkpointing():
    """Verify gradient checkpointing support"""
    model = ControlNetModel(
        conditioning_dim=512,
        gradient_checkpointing=True
    )
    assert model._supports_gradient_checkpointing
    assert any(hasattr(m, 'gradient_checkpointing') for m in model.modules())

def test_attention_processors():
    """Test attention processor handling"""
    model = ControlNetModel(conditioning_dim=512)
    
    # Test default processors
    processors = model.attn_processors
    assert len(processors) > 0
    
    # Test processor replacement
    new_processor = AttnProcessor()
    model.set_attn_processor(new_processor)
    assert all(isinstance(p, AttnProcessor) for p in model.attn_processors.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])