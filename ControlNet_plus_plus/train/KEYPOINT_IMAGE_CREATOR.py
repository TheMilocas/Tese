import os
import random
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from datasets import load_dataset

# ========== CONFIG ==========
DATASET_NAME = "Milocas/celebahq_clean"
OUTPUT_DIR = "OUTPUT_SINGLE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACEMESH_INDICES = {
    "eyes": [33, 133, 160, 159, 158, 157, 173, 246,
             362, 263, 387, 386, 385, 384, 398, 466],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405,
              321, 375, 291, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95, 185, 40, 39,
              37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78],
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5,
             4, 45, 220, 275, 440, 344, 278, 331]
}

# BGR colors for OpenCV
COLOR_MAP = {
    "eyes": (0, 255, 0),   # green
    "mouth": (0, 0, 255),  # red
    "nose": (255, 0, 0)    # blue
}

# ========== FUNCTIONS ==========
def create_rect_mask(image, landmarks, indices, padding_x=10, padding_y=10, scale=1.0):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Box center
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    bw, bh = (x_max - x_min), (y_max - y_min)

    # Expand box
    bw = int(bw * scale)
    bh = int(bh * scale)

    # Apply padding
    x_min = max(cx - bw // 2 - padding_x, 0)
    y_min = max(cy - bh // 2 - padding_y, 0)
    x_max = min(cx + bw // 2 + padding_x, w)
    y_max = min(cy + bh // 2 + padding_y, h)

    area = (x_max - x_min) * (y_max - y_min)
    return area, (x_min, y_min, x_max, y_max)

def draw_boxes_with_padding(image_np, landmarks, facemesh_indices, color_map, nose_eye_margin=30):
    """
    Draw bounding boxes and landmarks using the same padding logic
    as apply_mask_with_landmarks.
    """
    h, w, _ = image_np.shape
    vis_img = image_np.copy()

    # Compute lowest eye landmark Y
    eye_indices = facemesh_indices["eyes"]
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    lowest_eye_y = max([p[1] for p in eye_points])

    for region, indices in facemesh_indices.items():
        if region == "eyes":
            area, (x_min, y_min, x_max, y_max) = create_rect_mask(
                image_np, landmarks, indices, padding_x=0, padding_y=20, scale=1.5
            )

        elif region == "nose":
            area, (x_min, y_min, x_max, y_max) = create_rect_mask(
                image_np, landmarks, indices, padding_x=5, padding_y=5, scale=1
            )

            # Ensure nose does not overlap eyes
            if y_min < lowest_eye_y + nose_eye_margin:
                y_min = lowest_eye_y + nose_eye_margin
                area = (x_max - x_min) * (y_max - y_min)

        else:  # mouth
            area, (x_min, y_min, x_max, y_max) = create_rect_mask(
                image_np, landmarks, indices, padding_x=5, padding_y=5, scale=1
            )

        # Draw bounding box
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color_map[region], 2)

        # Draw landmarks
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        for (x, y) in points:
            cv2.circle(vis_img, (x, y), 2, color_map[region], -1)

        # Label
        cv2.putText(vis_img, region, (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[region], 2)

    return vis_img

# ========== MAIN ==========
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# Pick one random sample
idx = random.randint(0, len(dataset)-1)
sample = dataset[idx]
image = sample["image"].resize((512, 512))
image_np = np.array(image)  # RGB

# Detect landmarks and draw padded bounding boxes
with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(image_np)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        vis_img = draw_boxes_with_padding(image_np, landmarks, FACEMESH_INDICES, COLOR_MAP)

        # Show
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        # Save
        save_path = os.path.join(OUTPUT_DIR, f"sample_{idx}_with_padded_boxes.png")
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization: {save_path}")
    else:
        print("No face detected!")
