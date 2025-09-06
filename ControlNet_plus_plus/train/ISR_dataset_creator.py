import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ========== CONFIG ==========
INPUT_FOLDER = "./1"
OUTPUT_FOLDER = "./ISR_inputs"
TARGET_SIZE = (512, 512)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========== Initialize Face Detector ==========
app = FaceAnalysis(name='buffalo_l')  # uses RetinaFace + ArcFace
app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1)

def align_and_crop_face(image):
    faces = app.get(image)

    if len(faces) == 0:
        return None

    face = faces[0]
    bbox = face.bbox.astype(int)
    landmarks = face.kps  # 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]

    # Optional: do alignment using landmarks
    M, _ = cv2.estimateAffinePartial2D(
        np.array([landmarks[0], landmarks[1], landmarks[2]]),  # src: eyes + nose
        np.array([[180, 180], [332, 180], [256, 280]])  # dst: aligned positions
    )
    aligned = cv2.warpAffine(image, M, TARGET_SIZE, borderValue=(0, 0, 0))
    return aligned

# ========== Process Images ==========
image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

count = 1
for file_name in tqdm(image_files):
    path = os.path.join(INPUT_FOLDER, file_name)
    image = cv2.imread(path)

    if image is None:
        print(f"Could not read {file_name}")
        continue

    face_aligned = align_and_crop_face(image)
    if face_aligned is None:
        print(f"No face detected in {file_name}")
        continue

    output_path = os.path.join(OUTPUT_FOLDER, f"{count:03d}.png")
    cv2.imwrite(output_path, face_aligned)
    count += 1
