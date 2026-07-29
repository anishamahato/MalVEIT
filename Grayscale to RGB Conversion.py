#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# Input and Output Directories
# -----------------------------
input_dir = "C:/Users/Legion/malimg_dataset/malimg_paper_dataset_imgs"
output_dir = "C:/Users/Legion/malimg_dataset/malimg_rgb_trig"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# RGB Conversion Function
# -----------------------------
def grayscale_to_rgb_trig(gray_img):
    """
    Implements Algorithm 1:
    Grayscale -> Nonlinear RGB Transformation
    """

    # Equation (1): Normalize image
    gray = gray_img.astype(np.float32)

    I_min = gray.min()
    I_max = gray.max()

    if I_max == I_min:
        In = np.zeros_like(gray)
    else:
        In = (gray - I_min) / (I_max - I_min)

    # Equation (2)
    R = np.sin(np.pi * In)
    G = np.sin(np.pi * (In - 0.5))
    B = np.cos(np.pi * In)

    # Stack RGB channels
    rgb = np.stack([R, G, B], axis=-1)

    # Equation (3)
    rgb = 255 * ((rgb + 1) / 2)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


# -----------------------------
# Convert Complete Dataset
# -----------------------------
for family in os.listdir(input_dir):

    family_path = os.path.join(input_dir, family)

    if not os.path.isdir(family_path):
        continue

    save_family = os.path.join(output_dir, family)
    os.makedirs(save_family, exist_ok=True)

    images = os.listdir(family_path)

    for img_name in tqdm(images, desc=family):

        img_path = os.path.join(family_path, img_name)

        # Read grayscale image
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        # Apply Algorithm 1
        rgb = grayscale_to_rgb_trig(gray)

        # Save RGB image
        save_path = os.path.join(save_family, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

print("RGB Conversion Completed.")

