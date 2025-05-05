import os
import cv2
import numpy as np
from tqdm import tqdm

image_dir = 'datasets/celebahq/image'
mask_dir = 'datasets/celebahq/mask'
output_dir = 'datasets/celebahq/masked_images'

os.makedirs(output_dir, exist_ok=True)

image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(image_filenames):
    
    base_name = os.path.splitext(img_name)[0] 
    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, base_name + '.png') 
    
    if not os.path.exists(img_path):
        continue
    if not os.path.exists(mask_path):
        continue
        
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])
    masked_image = cv2.bitwise_and(image, cv2.bitwise_not(mask_3ch))

    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, masked_image)