from dataset_loader import CamouflageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

dataset = CamouflageDataset("mc_dataset/images", "mc_dataset/masks", size=256)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

image, mask = next(iter(loader))
img, mask = dataset[0]
print(img.shape, mask.shape)
print(torch.unique(mask))

print("Image tensor shape:", image.shape)
print("Mask tensor shape:", mask.shape)
print("Mask unique values:", mask.unique())

plt.subplot(1,2,1)
plt.imshow(image[0].permute(1,2,0))
plt.title("Resized Image")

plt.subplot(1,2,2)
plt.imshow(mask[0][0], cmap='gray')
plt.title("Binary Mask")

plt.show()