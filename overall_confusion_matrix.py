import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset_loader import CamouflageDataset
from models.sinet_gra import SINet_GRA


# =====================================================
# CONFIGURATION
# =====================================================
IMG_SIZE = 320
BATCH_SIZE = 4
THRESHOLD = 0.5

IMG_DIR = "combined_dataset/images"
MASK_DIR = "combined_dataset/masks"

CHECKPOINT_PATH = os.path.join(
    "checkpoints_combined",
    "sinet_gra_best.pth"
)

SAVE_DIR = "overall_confusion_matrix_results"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================================
# LOAD DATASET USING THE SAME SPLIT AS TRAINING
# =====================================================
full_dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=False
)

n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_idx, val_idx, test_idx = random_split(
    range(n_total),
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

test_set = torch.utils.data.Subset(
    full_dataset,
    test_idx.indices
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    f"Total: {n_total} | "
    f"Train: {n_train} | "
    f"Val: {n_val} | "
    f"Test: {len(test_set)}"
)


# =====================================================
# LOAD FINAL SINET + GRA MODEL
# =====================================================
model = SINet_GRA(pretrained=False).to(device)

try:
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
except TypeError:
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

model.load_state_dict(state_dict)
model.eval()

print("Final SINet + GRA model loaded successfully.")


# =====================================================
# ACCUMULATE CONFUSION MATRIX PIXELS
# =====================================================
overall_tn = 0
overall_fp = 0
overall_fn = 0
overall_tp = 0

processed_images = 0

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        predictions = (probs > THRESHOLD).long()
        ground_truth = (masks > 0.5).long()

        overall_tn += int(
            ((ground_truth == 0) & (predictions == 0)).sum().item()
        )

        overall_fp += int(
            ((ground_truth == 0) & (predictions == 1)).sum().item()
        )

        overall_fn += int(
            ((ground_truth == 1) & (predictions == 0)).sum().item()
        )

        overall_tp += int(
            ((ground_truth == 1) & (predictions == 1)).sum().item()
        )

        processed_images += imgs.shape[0]

        print(
            f"\rProcessed images: {processed_images}/{len(test_set)}",
            end=""
        )

print("\nTest-set pixel accumulation completed.")


# =====================================================
# CALCULATE PIXEL-LEVEL METRICS
# =====================================================
eps = 1e-7

total_pixels = overall_tn + overall_fp + overall_fn + overall_tp

accuracy = (
    (overall_tp + overall_tn)
    / (total_pixels + eps)
)

precision = (
    overall_tp
    / (overall_tp + overall_fp + eps)
)

recall = (
    overall_tp
    / (overall_tp + overall_fn + eps)
)

f1_score = (
    2 * precision * recall
    / (precision + recall + eps)
)

dice_score = (
    2 * overall_tp
    / (2 * overall_tp + overall_fp + overall_fn + eps)
)

iou_score = (
    overall_tp
    / (overall_tp + overall_fp + overall_fn + eps)
)

specificity = (
    overall_tn
    / (overall_tn + overall_fp + eps)
)

false_positive_rate = (
    overall_fp
    / (overall_fp + overall_tn + eps)
)

false_negative_rate = (
    overall_fn
    / (overall_fn + overall_tp + eps)
)


# =====================================================
# PRINT RESULTS
# =====================================================
print("\n========== OVERALL PIXEL-LEVEL CONFUSION MATRIX ==========")
print(f"True Negative  (TN): {overall_tn:,}")
print(f"False Positive (FP): {overall_fp:,}")
print(f"False Negative (FN): {overall_fn:,}")
print(f"True Positive  (TP): {overall_tp:,}")
print(f"Total Pixels       : {total_pixels:,}")

print("\n========== OVERALL PIXEL-LEVEL METRICS ==========")
print(f"Accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision   : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall      : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score    : {f1_score:.4f} ({f1_score * 100:.2f}%)")
print(f"Dice Score  : {dice_score:.4f} ({dice_score * 100:.2f}%)")
print(f"IoU         : {iou_score:.4f} ({iou_score * 100:.2f}%)")
print(f"Specificity : {specificity:.4f} ({specificity * 100:.2f}%)")
print(f"FPR         : {false_positive_rate:.4f}")
print(f"FNR         : {false_negative_rate:.4f}")


# =====================================================
# CREATE OVERALL CONFUSION MATRIX PNG
# =====================================================
confusion_matrix = np.array([
    [overall_tn, overall_fp],
    [overall_fn, overall_tp]
], dtype=np.int64)

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(confusion_matrix, cmap="Blues")

ax.set_title(
    "Overall Pixel-Level Confusion Matrix\n"
    f"Final SINet + GRA Model ({len(test_set)} Test Images)",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("Ground Truth Class", fontsize=12)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Background", "Target"])
ax.set_yticklabels(["Background", "Target"])

cell_names = [
    ["TN", "FP"],
    ["FN", "TP"]
]

maximum_value = confusion_matrix.max()

for row in range(2):
    for column in range(2):
        value = confusion_matrix[row, column]

        text_color = (
            "white"
            if value > maximum_value / 2
            else "black"
        )

        ax.text(
            column,
            row - 0.05,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=text_color
        )

        ax.text(
            column,
            row + 0.18,
            cell_names[row][column],
            ha="center",
            va="center",
            fontsize=11,
            color=text_color
        )

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Number of Pixels", fontsize=11)

plt.tight_layout()

matrix_path = os.path.join(
    SAVE_DIR,
    "overall_pixel_confusion_matrix.png"
)

plt.savefig(
    matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved overall confusion matrix:")
print(matrix_path)


# =====================================================
# SAVE RESULTS TO CSV
# =====================================================
metrics_path = os.path.join(
    SAVE_DIR,
    "overall_pixel_metrics.csv"
)

with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(["Evaluation Setting", "Value"])
    writer.writerow(["Model", "Adapted SINet + GRA"])
    writer.writerow(["Test Images", len(test_set)])
    writer.writerow(["Image Size", f"{IMG_SIZE}x{IMG_SIZE}"])
    writer.writerow(["Threshold", THRESHOLD])

    writer.writerow([])
    writer.writerow(["Confusion Matrix Component", "Pixel Count"])
    writer.writerow(["True Negative", overall_tn])
    writer.writerow(["False Positive", overall_fp])
    writer.writerow(["False Negative", overall_fn])
    writer.writerow(["True Positive", overall_tp])
    writer.writerow(["Total Pixels", total_pixels])

    writer.writerow([])
    writer.writerow(["Metric", "Decimal", "Percentage"])
    writer.writerow(["Accuracy", accuracy, accuracy * 100])
    writer.writerow(["Precision", precision, precision * 100])
    writer.writerow(["Recall", recall, recall * 100])
    writer.writerow(["F1-Score", f1_score, f1_score * 100])
    writer.writerow(["Dice Score", dice_score, dice_score * 100])
    writer.writerow(["IoU", iou_score, iou_score * 100])
    writer.writerow(["Specificity", specificity, specificity * 100])
    writer.writerow(["False Positive Rate", false_positive_rate, false_positive_rate * 100])
    writer.writerow(["False Negative Rate", false_negative_rate, false_negative_rate * 100])

print("Saved overall metrics:")
print(metrics_path)


# =====================================================
# SAVE RESULTS TO TEXT FILE
# =====================================================
text_path = os.path.join(
    SAVE_DIR,
    "overall_pixel_results.txt"
)

with open(text_path, "w", encoding="utf-8") as text_file:
    text_file.write(
        "OVERALL PIXEL-LEVEL RESULTS\n"
        "===========================\n\n"
    )

    text_file.write("Model: Adapted SINet + GRA\n")
    text_file.write(f"Test images: {len(test_set)}\n")
    text_file.write(f"Threshold: {THRESHOLD}\n\n")

    text_file.write("Confusion Matrix:\n")
    text_file.write(f"TN: {overall_tn:,}\n")
    text_file.write(f"FP: {overall_fp:,}\n")
    text_file.write(f"FN: {overall_fn:,}\n")
    text_file.write(f"TP: {overall_tp:,}\n")
    text_file.write(f"Total pixels: {total_pixels:,}\n\n")

    text_file.write("Metrics:\n")
    text_file.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    text_file.write(f"Precision: {precision:.4f} ({precision * 100:.2f}%)\n")
    text_file.write(f"Recall: {recall:.4f} ({recall * 100:.2f}%)\n")
    text_file.write(f"F1-Score: {f1_score:.4f} ({f1_score * 100:.2f}%)\n")
    text_file.write(f"Dice Score: {dice_score:.4f} ({dice_score * 100:.2f}%)\n")
    text_file.write(f"IoU: {iou_score:.4f} ({iou_score * 100:.2f}%)\n")
    text_file.write(f"Specificity: {specificity:.4f} ({specificity * 100:.2f}%)\n")

print("Saved text summary:")
print(text_path)

print(f"\nAll results saved in: {SAVE_DIR}/")