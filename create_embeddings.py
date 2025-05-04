import os
import cv2
import numpy as np
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from tqdm import tqdm 

model = ArcFaceONNX(model_file='ARCFACE/models/glintr100.onnx')

def get_embedding(image_path):
    """Extract normalized face embedding using ArcFace ONNX"""
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    embedding = model.get_feat(img)[0]

    return embedding / np.linalg.norm(embedding)

def process_folder(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        embedding = get_embedding(img_path)
        
        np.save(
            os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy'),
            embedding.astype(np.float32)  
        )

process_folder(
    image_dir='datasets/celebahq/image', 
    output_dir='datasets/celebahq/conditions'
)