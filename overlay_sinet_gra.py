import os
import torch
import numpy as np
import cv2
from torch.utils.data import random_split

from dataset_loader import CamouflageDataset
from models.sinet_gra import SINet_GRA

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = 320
THRESHOLD = 0.4

IMG_DIR = "combined_dataset/images"
MASK_DIR = "combined_dataset/masks"
CHECKPOINT_PATH = "checkpoints_combined/sinet_gra_best.pth"

SAVE_DIR = "overlay_selected_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Choose selected test images
# These follow the same test set order as your confusion matrix code
SELECTED_TEST_INDICES = [0, 1, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# DATASET WITH SAME SPLIT AS FINAL TRAINING
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
# LOAD FINAL MODEL
# =====================================================
model = SINet_GRA(pretrained=False).to(device)

try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)

model.load_state_dict(state_dict)
model.eval()

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_file_name_from_dataset(dataset, original_index):
    """
    Try to retrieve image file name from dataset_loader.
    If not available, use sorted image folder as fallback.
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

    # fallback: assumes dataset uses sorted image files
    image_files = sorted(os.listdir(IMG_DIR))
    return image_files[original_index]


def denormalize_image(img_tensor):
    """
    Convert normalized tensor image back to RGB image.
    This follows ImageNet mean/std used during training.
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def create_overlay_with_gt_contour(img_np, gt, pred):
    """
    Overlay explanation:
    - Green area = predicted target mask
    - Red outline = ground truth target boundary
    """

    overlay = img_np.copy()

    # Green transparent prediction overlay
    pred_region = pred == 1
    green = np.array([0, 255, 0], dtype=np.uint8)

    overlay[pred_region] = (
        0.55 * overlay[pred_region] + 0.45 * green
    ).astype(np.uint8)

    # Red ground truth contour
    gt_uint8 = (gt * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        gt_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw red contour on overlay
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(
        overlay_bgr,
        contours,
        -1,
        (0, 0, 255),   # red in BGR
        2
    )

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb


# =====================================================
# GENERATE OVERLAY IMAGES FOR SELECTED TEST IMAGES
# =====================================================
for sample_no, test_index in enumerate(SELECTED_TEST_INDICES, start=1):

    original_dataset_index = test_idx.indices[test_index]
    image_file_name = get_file_name_from_dataset(dataset, original_dataset_index)
    clean_name = os.path.splitext(image_file_name)[0]

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

    img_np = denormalize_image(img)
    overlay = create_overlay_with_gt_contour(img_np, gt, pred)

    # Save original image
    cv2.imwrite(
        os.path.join(SAVE_DIR, f"sample_{sample_no}_{clean_name}_original.png"),
        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    )

    # Save ground truth mask
    cv2.imwrite(
        os.path.join(SAVE_DIR, f"sample_{sample_no}_{clean_name}_ground_truth.png"),
        gt * 255
    )

    # Save predicted mask
    cv2.imwrite(
        os.path.join(SAVE_DIR, f"sample_{sample_no}_{clean_name}_predicted_mask.png"),
        pred * 255
    )

    # Save overlay image
    cv2.imwrite(
        os.path.join(SAVE_DIR, f"sample_{sample_no}_{clean_name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print("Saved overlay result.")

print(f"\nAll overlay images saved in: {SAVE_DIR}/")