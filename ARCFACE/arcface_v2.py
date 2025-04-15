import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from PIL import Image
from tqdm import tqdm
import random
import cv2
import json
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from insightface.model_zoo.arcface_onnx import ArcFaceONNX

model = ArcFaceONNX(model_file='models/glintr100.onnx')
model.prepare(ctx_id=0)

def get_embedding(image_path):
    img = cv2.imread(image_path)    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    embedding = model.get_feat(img_bgr)

    embedding = np.array(embedding).squeeze()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return np.zeros_like(embedding)
    return embedding / norm

def intra_person_analysis(id2image, data_root, selected_ids):
    distances = []
    for pid in tqdm(selected_ids, desc="Intra-person analysis"):
        image_names = id2image[pid]['ref'] + id2image[pid]['infer']
        embeddings = []
        for img_name in image_names:
            img_path = os.path.join(data_root, img_name)
            emb = get_embedding(img_path)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)
        pairwise = cosine_distances(embeddings)
        dist_values = pairwise[np.triu_indices(len(embeddings), k=1)]
        distances.append({
            'person_id': pid,
            'mean_dist': np.mean(dist_values),
            'std': np.std(dist_values),
            'min': np.min(dist_values),
            'max': np.max(dist_values),
        })
    return distances

def inter_person_analysis(id2image, data_root, selected_id_pairs):
    distances = []
    for id1, id2 in tqdm(selected_id_pairs, desc="Inter-person analysis"):
        img1 = id2image[id1]['ref'][0]  # or random.choice(...)
        img2 = id2image[id2]['ref'][0]

        emb1 = get_embedding(os.path.join(data_root, img1))
        emb2 = get_embedding(os.path.join(data_root, img2))

        dist = cosine_distances([emb1], [emb2])[0][0]
        distances.append({
            'id1': id1,
            'id2': id2,
            'distance': dist
        })
    return distances

def plot_roc_curve(intra_dists, inter_dists):

    y_true = [1]*len(intra_dists) + [0]*len(inter_dists)
    y_score = intra_dists + inter_dists  # lower = more similar
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Verification ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

def plot_arcface_style_hist(intra_dists, inter_dists):
    plt.figure(figsize=(8, 6))
    
    sns.histplot(inter_dists, bins=80, color='blue', label='Inter-person', kde=False, stat="count", element="step")
    sns.histplot(intra_dists, bins=80, color='red', label='Intra-person', kde=False, stat="count", element="step")
    
    plt.title("Cosine Distance Distribution\nIntra- vs Inter-person Embeddings")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Number of Pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig('arcface_style_hist_2.png')
    plt.close()

# Load data
with open('celebahq-idi/celebahq/annotation/idi-20.json', 'r') as f:
    dataset = json.load(f)

id2image = dataset['id2image']
data_root = 'celebahq-idi/celebahq/image'

# Pick smaller number of IDs
selected_ids = random.sample(list(id2image.keys()), 20)

# Generate ID pairs
selected_pairs = list(combinations(selected_ids, 2))
random.shuffle(selected_pairs)
selected_pairs = selected_pairs[:500]

# Analysis
intra_results = intra_person_analysis(id2image, data_root, selected_ids)
inter_results = inter_person_analysis(id2image, data_root, selected_pairs)

intra_dists = [r['mean_dist'] for r in intra_results]
inter_dists = [r['distance'] for r in inter_results]

plot_arcface_style_hist(intra_dists, inter_dists)
#plot_roc_curve(intra_dists, inter_dists)