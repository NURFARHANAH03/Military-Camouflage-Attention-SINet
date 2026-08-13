import os
import json

from dataset_loader import CamouflageDataset
from group_split import create_group_split

IMG_SIZE = 320
SPLIT_SEED = 42

IMG_DIR = "combined_dataset/images"
MASK_DIR = "combined_dataset/masks"
SAVE_DIR = "checkpoints_combined"

os.makedirs(SAVE_DIR, exist_ok=True)

full_dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=False
)

train_indices, val_indices, test_indices = create_group_split(
    image_files=full_dataset.images,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=SPLIT_SEED
)

split_info = {
    "split_seed": SPLIT_SEED,
    "dataset": "combined_dataset",
    "train": [
        full_dataset.images[i]
        for i in train_indices
    ],
    "validation": [
        full_dataset.images[i]
        for i in val_indices
    ],
    "test": [
        full_dataset.images[i]
        for i in test_indices
    ]
}

split_path = os.path.join(
    SAVE_DIR,
    f"combined_dataset_split_seed{SPLIT_SEED}.json"
)

with open(split_path, "w", encoding="utf-8") as file:
    json.dump(split_info, file, indent=4)

print("\n========== SAVED DATASET SPLIT ==========")
print("Split seed :", SPLIT_SEED)
print("Train      :", len(split_info["train"]))
print("Validation :", len(split_info["validation"]))
print("Test       :", len(split_info["test"]))
print("Saved to  :", split_path)