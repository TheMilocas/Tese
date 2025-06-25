import os
import cv2
import numpy as np
import face_alignment
from datasets import load_dataset
from skimage import img_as_ubyte
from skimage.io import imsave
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
dataset_name = "Milocas/celebahq_clean"
output_dir = "categorized_masks_debug"
debug_dir = "mask_debug_visuals"
split = "test"
threshold_coverage = 0.3
threshold_precision = 0.1
max_mask_ratio = 0.4  # optional: skip masks that are too large

os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

dataset = load_dataset(dataset_name, split=split)

fa = face_alignment.FaceAlignment('2D', flip_input=False)

def get_roi_bbox(landmarks, indices, margin=5):
    pts = landmarks[indices]
    xmin, ymin = np.floor(pts.min(axis=0)).astype(int) - margin
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(int) + margin
    return max(xmin, 0), max(ymin, 0), xmax, ymax

def evaluate_occlusion(mask, bbox):
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0, 0.0

    mask_area = np.count_nonzero(mask)
    region_area = region.size
    region_covered = np.count_nonzero(region)

    if mask_area == 0:
        return 0.0, 0.0

    coverage_in_roi = region_covered / region_area
    precision_of_mask = region_covered / mask_area

    return coverage_in_roi, precision_of_mask

for idx, example in enumerate(tqdm(dataset)):
    image = example["image"]
    mask = example["mask"]

    img = np.array(image)
    m_np = np.array(mask)

    m = cv2.cvtColor(m_np, cv2.COLOR_RGB2GRAY) if m_np.ndim == 3 else m_np

    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Optional: skip overly large masks
    if np.count_nonzero(m) / m.size > max_mask_ratio:
        continue
        
    try:
        landmarks = fa.get_landmarks(img)[0]
    except TypeError:
        continue  # skip images with no detected face

    rois = {
        "eyes": list(range(36, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68))
    }

    cats = []
    debug_img = img.copy()
    for name, indices in rois.items():
        bbox = get_roi_bbox(landmarks, indices)
        x0, y0, x1, y1 = bbox

        # Evaluate mask coverage & precision
        coverage, precision = evaluate_occlusion(m, bbox)
        # print(f"[{idx}] {name} - coverage: {coverage:.2f}, precision: {precision:.2f}")

        # Draw bbox
        color = {'eyes': (255, 255, 0), 'nose': (0, 255, 255), 'mouth': (255, 0, 255)}[name]
        cv2.rectangle(debug_img, (x0, y0), (x1, y1), color, 2)

        # Check criteria
        if coverage >= threshold_coverage and precision >= threshold_precision:
            cats.append(name)

    # Overlay mask on debug image (red)
    debug_img[m > 0] = [255, 0, 0]

    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(debug_img, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Save debug image
    debug_out_path = os.path.join(debug_dir, f"{idx:06d}.png")
    cv2.imwrite(debug_out_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    if not cats:
        continue

    # Save mask in categorized folder
    category_name = "_".join(sorted(cats))
    out_dir = os.path.join(output_dir, category_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{idx:06d}.png")
    imsave(out_path, img_as_ubyte(m))
