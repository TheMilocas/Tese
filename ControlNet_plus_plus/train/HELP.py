import os, sys, torch, glob
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn

sys.path.append("/home/csantiago/ControlNet_plus_plus/train/diffusers_new/src")
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from backbones.iresnet import iresnet100

sys.path.append("/home/csantiago/PVA-CelebAHQ-IDI/lib")
from dataset import CelebAHQIDIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./COMPARE_PVA/output_base_new", exist_ok=True)
os.makedirs("./COMPARE_PVA/output_controlnet_new", exist_ok=True)

# --- ArcFace wrapper ---
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess_image(image_pil):
    return transform(image_pil).unsqueeze(0).to(device)

# --- Load pipelines ---
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_model_cpu_offload()

CONTROLNET_PATH = "../../identity_controlnet_final"
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

# --- ArcFace ---
arcface_model_path = os.path.abspath("../../ARCFACE/models/R100_MS1MV3/backbone.pth")
arcface = ARCFACE(arcface_model_path, device=device).to(device)

# --- Dataset ---
dataset_path = "/home/csantiago/PVA-CelebAHQ-IDI/data/celebahq"
test_ds = CelebAHQIDIDataset(
    data_dir=dataset_path,
    split="test",
    use_caption=False,
    inpaint_region=["lowerface", "eyebrow", "wholeface"],
    seed=2023
)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

# --- Run inference ---
for batch in tqdm(test_dl):
    infer_images = batch["infer_image"]  # (1, N_inf, 3, H, W)
    masks        = batch["infer_mask"]   # (1, N_inf, num_mask, 1, H, W)
    ref_images   = batch["ref_image"]    # (1, N_REF, 3, H, W)
    all_files    = batch["all_file"]     # list of filenames

    # Remove batch dim for easier indexing
    infer_images = infer_images[0]  # (N_inf, 3, H, W)
    masks        = masks[0]         # (N_inf, num_mask, 1, H, W)
    ref_images   = ref_images[0]    # (N_REF, 3, H, W)

    # ArcFace embeddings for ControlNet
    emb_batch = []
    for img_tensor in infer_images:
        img_pil = transforms.ToPILImage()(img_tensor)
        emb = arcface(preprocess_image(img_pil))
        emb_batch.append(emb)
    emb_batch = torch.cat(emb_batch, 0)  # (N_inf, 512)

    # Iterate over images and masks
    for i in range(len(infer_images)):  # iterate over the actual number of images
        img_tensor = infer_images[i]    # (3, H, W)
        fname = all_files[i][0] if isinstance(all_files[i], tuple) else all_files[i]
        img_pil = transforms.ToPILImage()(img_tensor)

        num_masks = masks.shape[1]  # number of masks for this image
        for mask_idx in range(num_masks):
            mask_tensor = masks[i, mask_idx]  # (1, H, W) or (1,1,H,W)
            # Convert mask to PIL, handling channel dim
            mask_pil = transforms.ToPILImage()(mask_tensor.squeeze(0)).convert("L")

            id_val = int(batch["id"][0])  
            image_idx = int(os.path.splitext(all_files[i][0])[0]) 
            filename = f"{id_val}_{image_idx}_{mask_idx}.jpg"

            base_path = f"./COMPARE_PVA/output_base_new/{filename}"
            controlnet_path = f"./COMPARE_PVA/output_controlnet_new/{filename}"

            # --- Check if both outputs already exist ---
            if os.path.exists(base_path) and os.path.exists(controlnet_path):
                print(f"Skipping {filename}, already exists.")
                continue

            # --- Vanilla SD ---
            if not os.path.exists(base_path):
                result_base = pipe_base(
                    prompt="",
                    image=img_pil,
                    mask_image=mask_pil,
                    num_inference_steps=100,
                    generator=torch.Generator(device).manual_seed(2023)
                ).images[0]
                result_base.save(base_path)

            # --- ControlNet SD ---
            if not os.path.exists(controlnet_path):
                result_controlnet = pipe_controlnet(
                    prompt="",
                    image=img_pil,
                    mask_image=mask_pil,
                    control_image=emb_batch[i].unsqueeze(0),
                    controlnet_conditioning_scale=1.0,
                    num_inference_steps=100,
                    generator=torch.Generator(device).manual_seed(2023)
                ).images[0]
                result_controlnet.save(controlnet_path)
