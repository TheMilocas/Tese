import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import random

from sklearn.datasets import fetch_lfw_people
from sklearn.metrics.pairwise import cosine_similarity

from insightface.model_zoo.arcface_onnx import ArcFaceONNX

lfw = fetch_lfw_people(color=True, resize=1.0)
NUM_IMAGES = 100
indices = random.sample(range(len(lfw.images)), NUM_IMAGES)
selected_images = [lfw.images[i] for i in indices]
selected_labels = [lfw.target[i] for i in indices]

# os.makedirs("sample_lfw_images", exist_ok=True)
# for i in range(min(10, len(selected_images))):
#     img = (selected_images[i] * 255).astype(np.uint8)
#     Image.fromarray(img).save(f"sample_lfw_images/lfw_sample_{i}.png")

model = ArcFaceONNX(model_file='models/glintr100.onnx')
model.prepare(ctx_id=0)

embeddings = []
valid_labels = []

print("Extracting embeddings:")
for img, label in tqdm(zip(selected_images, selected_labels), total=NUM_IMAGES):
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (112, 112))
    embedding = model.get_feat(img_resized)
    embeddings.append(embedding)
    valid_labels.append(label)

embeddings = np.array(embeddings)
if embeddings.ndim > 2:
    embeddings = embeddings.squeeze()

same_person_sims = []
different_person_sims = []

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]
        if valid_labels[i] == valid_labels[j]:
            same_person_sims.append(sim)
        else:
            different_person_sims.append(sim)

plt.hist(same_person_sims, bins=50, alpha=0.6, label='Same Person')
plt.hist(different_person_sims, bins=50, alpha=0.6, label='Different People')
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("Cosine Similarity: Same vs Different People")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("same_vs_diff_hist.png")

# Save some image pairs for inspection
os.makedirs("same_person_pairs", exist_ok=True)
pair_count = 0
threshold = 0.5

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        if valid_labels[i] == valid_labels[j]:
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[j].reshape(1, -1)
            )[0][0]
            if sim > threshold and pair_count < 10:
                img1 = (selected_images[i] * 255).astype(np.uint8)
                img2 = (selected_images[j] * 255).astype(np.uint8)
                combined = np.hstack((img1, img2))
                Image.fromarray(combined).save(f"same_person_pairs/pair_{pair_count}_sim_{sim:.2f}.png")
                pair_count += 1
