import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from dataset_loader import CamouflageDataset
from models.sinet_gra import SINet_GRA

#only selected images

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = 320
THRESHOLD = 0.4

IMG_DIR = "combined_dataset/images"
MASK_DIR = "combined_dataset/masks"
CHECKPOINT_PATH = "checkpoints_combined/sinet_gra_best.pth"

SAVE_DIR = "confusion_matrix_selected_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Choose selected test images
# Example: first 3 images from test set
SELECTED_TEST_INDICES = [0, 1, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LOAD DATASET USING SAME SPLIT AS FINAL TRAINING
# =====================================================
dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=False
)

n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_idx, val_idx, test_idx = random_split(
    range(n_total),
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

test_set = torch.utils.data.Subset(dataset, test_idx.indices)

print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

# =====================================================
# LOAD FINAL SINet + GRA MODEL
# =====================================================
model = SINet_GRA(pretrained=False).to(device)

try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)

model.load_state_dict(state_dict)
model.eval()

# =====================================================
# CONFUSION MATRIX FUNCTIONS
# =====================================================
def calculate_confusion_matrix(pred, gt):
    """
    Pixel-level confusion matrix for binary segmentation.

    Class 0 = Background
    Class 1 = Camouflaged target

    Matrix format:
    [[TN, FP],
     [FN, TP]]
    """

    pred = pred.flatten()
    gt = gt.flatten()

    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tp = np.sum((gt == 1) & (pred == 1))

    return np.array([[tn, fp],
                     [fn, tp]])


def plot_confusion_matrix(cm, save_path, title):
    labels = ["Background", "Target"]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)

    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    # Write values inside cells
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black"
            )

    # Add cell labels
    plt.text(0, 0.25, "TN", ha="center", va="center", fontsize=10)
    plt.text(1, 0.25, "FP", ha="center", va="center", fontsize=10)
    plt.text(0, 1.25, "FN", ha="center", va="center", fontsize=10)
    plt.text(1, 1.25, "TP", ha="center", va="center", fontsize=10)

    plt.colorbar(label="Number of Pixels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =====================================================
# GENERATE CONFUSION MATRIX IMAGES
# =====================================================
def get_file_name_from_dataset(dataset, original_index):
    """
    Try to get the image file name from CamouflageDataset.
    This depends on how your dataset_loader.py stores image paths.
    """

    possible_attrs = [
        "image_paths",
        "img_paths",
        "images",
        "image_files",
        "img_files"
    ]

    for attr in possible_attrs:
        if hasattr(dataset, attr):
            file_list = getattr(dataset, attr)
            file_path = file_list[original_index]
            return os.path.basename(file_path)

    return f"dataset_index_{original_index}"

for sample_no, test_index in enumerate(SELECTED_TEST_INDICES, start=1):

        # Get original index from the full dataset
        original_dataset_index = test_idx.indices[test_index]

        # Get file name
        image_file_name = get_file_name_from_dataset(dataset, original_dataset_index)

        print(f"\nSample {sample_no}")
        print(f"Test set index: {test_index}")
        print(f"Original dataset index: {original_dataset_index}")
        print(f"Image file name: {image_file_name}")

        img, mask = test_set[test_index]

        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_input)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred = (prob > THRESHOLD).astype(np.uint8)
        gt = (mask[0].numpy() > 0.5).astype(np.uint8)

        cm = calculate_confusion_matrix(pred, gt)

        # Clean file name for saving
        clean_name = os.path.splitext(image_file_name)[0]

        save_path = os.path.join(
            SAVE_DIR,
            f"sample_{sample_no}_{clean_name}_confusion_matrix.png"
        )

        plot_confusion_matrix(
            cm,
            save_path,
            title=f"Pixel-Level Confusion Matrix\n{image_file_name}"
        )

        print("Confusion Matrix:")
        print(cm)
        print("Saved:", save_path)   

print(f"\nAll confusion matrix images saved in: {SAVE_DIR}/")