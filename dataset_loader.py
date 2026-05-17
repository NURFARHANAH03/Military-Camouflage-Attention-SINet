import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CamouflageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.size = size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(size, size),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Fix mismatched image and mask size
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"].float() / 255.0          # (3,H,W)
        mask = augmented["mask"]                            # (H,W)

        # Convert mask to binary tensor (1,H,W)
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask