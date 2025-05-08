# from diffusers import ControlNetModel
# import torch
# from safetensors.torch import save_file
# import os

# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-canny",
#     torch_dtype=torch.float32
# )

# for param in controlnet.parameters():
#     param.data.zero_()

# output_dir = "zeroed_controlnet"
# os.makedirs(output_dir, exist_ok=True)

# save_file(controlnet.state_dict(), os.path.join(output_dir, "diffusion_pytorch_model.safetensors"))

# print(f"Zeroed ControlNet folder created at: {output_dir}")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

def zero_out_controlnet(input_path, output_path):
    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    zero_tensors = {k: torch.zeros_like(v) for k, v in tensors.items()}

    save_file(zero_tensors, output_path)
    print(f"Saved zeroed-out weights to {output_path}")

input_file = "trained_controlnet_inicial/diffusion_pytorch_model.safetensors"
output_file = "zeroed_controlnet.safetensors"
zero_out_controlnet(input_file, output_file)