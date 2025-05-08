from diffusers import ControlNetModel
import torch
from safetensors.torch import save_file
import os

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32
)

for param in controlnet.parameters():
    param.data.zero_()

output_dir = "zeroed_controlnet"
os.makedirs(output_dir, exist_ok=True)

save_file(controlnet.state_dict(), os.path.join(output_dir, "diffusion_pytorch_model.safetensors"))

print(f"Zeroed ControlNet folder created at: {output_dir}")