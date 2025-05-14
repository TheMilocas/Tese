from PIL import Image
import torch
from transformers import AutoTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load trained model
controlnet = ControlNetModel.from_pretrained("diffusers_new/src/diffusers/models/controlnets/controlnet")
tokenizer = AutoTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

# Initialize pipeline with your custom arguments
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Test with a known embedding
test_embedding = torch.load("test_embedding.pt")  # Your reference embedding
generated_image = pipe(
    "a person smiling", 
    identity_embedding=test_embedding,
    num_inference_steps=20
).images[0]

Image.show(generated_image)