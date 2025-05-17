from diffusers import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel  

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
controlnet = ControlNetModel.from_unet(unet, load_weights_from_unet=True)
