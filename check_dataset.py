import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Change this path if needed
image_dir = "mc_dataset/images"
mask_dir = "mc_dataset/masks"

# Pick random image
import random

for i in range(2):
    image_name = random.choice(sorted(os.listdir(image_dir)))
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, image_name.replace(".jpg", ".png"))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, 0)

    print("Sample:", image_name)
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Unique mask values:", np.unique(mask))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.show()

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load mask (grayscale)
mask = cv2.imread(mask_path, 0)

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Unique mask values:", np.unique(mask))

# Visualize
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")

plt.show()