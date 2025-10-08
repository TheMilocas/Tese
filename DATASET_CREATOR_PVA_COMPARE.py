import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys, os
sys.path.append("/home/csantiago/PVA-CelebAHQ-IDI")  # <-- adjust if repo is elsewhere

from lib.dataset import CelebAHQIDIDataset
from backbones.iresnet import iresnet100

# ---------------- Config -----------------
dataset_dir = "/home/csantiago/PVA-CelebAHQ-IDI/CelebAHQ-IDI"
manifest = "/home/csantiago/PVA-CelebAHQ-IDI/588x4_manifest.csv"
arcface_ckpt = "/home/csantiago/ARCFACE/models/R100_MS1MV3/backbone.pth"
outdir = "/home/csantiago/precomputed_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 2023
torch.manual_seed(seed)

os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
os.makedirs(os.path.join(outdir, "masks"), exist_ok=True)
os.makedirs(os.path.join(outdir, "embeddings"), exist_ok=True)

# ---------------- Dataset -----------------
ds = CelebAHQIDIDataset(
    data_dir=dataset_dir,
    split="test",
    use_caption=False,
    inpaint_region=["lowerface", "eyebrow", "wholeface"],
    seed=seed,
    loop_data="identity"
)

# ---------------- Manifest ----------------
rows = []
with open(manifest, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

id_to_ds_index = {int(v): i for i, v in enumerate(ds.ids)}

# ---------------- ArcFace -----------------
class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(ARCFACE, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface = ARCFACE(arcface_ckpt, device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def compute_arcface_embedding(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    emb = arcface(x)  # shape [1, 512]
    return emb.squeeze(0).cpu()

# ---------------- Helpers -----------------
def tensor_to_pil(img_t):
    return transforms.ToPILImage()(img_t.cpu())

def extract_image(item, infer_index):
    return tensor_to_pil(item["infer_image"][infer_index])

def extract_mask(item, infer_index, mask_index):
    imask = item["infer_mask"]
    num_mask_types = imask.shape[1] if imask.dim() >= 4 else 1

    if mask_index < num_mask_types:
        if imask.dim() == 5:
            m = imask[infer_index, mask_index, 0]
        else:
            m = imask[infer_index, mask_index]
    else:
        # print(f"Mask index {mask_index} out of range (only {num_mask_types}), using 0 instead")
        if imask.dim() == 5:
            m = imask[infer_index, 0, 0]
        else:
            m = imask[infer_index, 0]

    m = (m.clamp(0,1) * 255).byte().cpu()
    return Image.fromarray(m.numpy(), mode="L")

# ---------------- Main Loop -----------------
for row in tqdm(rows, desc="Processing dataset"):
    idn = int(row["id"])
    image_id = row["image_id"]
    mask_idx = int(row["mask_index"])

    if idn not in id_to_ds_index:
        print("id not found:", idn)
        continue

    item = ds[id_to_ds_index[idn]]

    # match infer index
    infer_idx = 0
    for i, fname in enumerate(item["all_file"]):
        base = os.path.splitext(fname)[0]
        if base == image_id:
            infer_idx = i
            break

    try:
        image_pil = extract_image(item, infer_idx)
        mask_pil = extract_mask(item, infer_idx, mask_idx)
    except Exception as e:
        print("Error extracting:", e)
        continue

    # save image + mask
    image_path = os.path.join(outdir, "images", f"{idn}_{image_id}.png")
    mask_path = os.path.join(outdir, "masks", f"{idn}_{image_id}_mask{mask_idx}.png")
    image_pil.save(image_path)
    mask_pil.save(mask_path)

    # compute and save embedding
    embedding = compute_arcface_embedding(image_pil)
    torch.save(embedding, os.path.join(outdir, "embeddings", f"{idn}_{image_id}.npy"))
