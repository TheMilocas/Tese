import torch
import torch.nn as nn
from torchvision import transforms
from insightface.model_zoo import get_model, model_zoo

# Load IR-SE-ResNet100 backbone
model = get_model('model.onnx', download=True)


# Confirm it's adaptable to 512×512
dummy_input = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    output = model(dummy_input)
print("Output shape:", output.shape)  # Should be (1, 512)
