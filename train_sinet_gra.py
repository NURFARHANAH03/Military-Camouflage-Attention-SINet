import os
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader

from dataset_loader import CamouflageDataset
from group_split import create_group_split
from models.sinet_gra import SINet_GRA

# =========================================================
# Experiment 2: Final SINet + GRA on Combined Dataset
# =========================================================

IMG_SIZE = 320
BATCH_SIZE = 4
PIN_MEMORY = True
EPOCHS = 30
LR = 5e-5

SPLIT_SEED = 42
TRAIN_SEED = 42

IMG_DIR = "combined_dataset/images"
MASK_DIR = "combined_dataset/masks"
SAVE_DIR = "checkpoints_combined"
PRED_DIR = "pred_samples_sinet_gra_combined"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    SAVE_DIR,
    f"sinet_gra_best_seed{TRAIN_SEED}.pth"
)

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(TRAIN_SEED)
random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Split seed   : {SPLIT_SEED}")
print(f"Training seed: {TRAIN_SEED}")

# Dataset + group-based split
full_dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=False
)

n_total = len(full_dataset)

train_indices, val_indices, test_indices = create_group_split(
    image_files=full_dataset.images,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=SPLIT_SEED
)

# =========================================================
# Save exact dataset split for GUI and reproducibility
# =========================================================
split_info = {
    "split_seed": SPLIT_SEED,
    "training_seed": TRAIN_SEED,
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

SPLIT_FILE_PATH = os.path.join(
    SAVE_DIR,
    f"combined_dataset_split_seed{SPLIT_SEED}.json"
)

with open(SPLIT_FILE_PATH, "w", encoding="utf-8") as file:
    json.dump(split_info, file, indent=4)

print("Saved dataset split:", SPLIT_FILE_PATH)
print("Saved training filenames :", len(split_info["train"]))
print("Saved validation filenames:", len(split_info["validation"]))
print("Saved test filenames      :", len(split_info["test"]))

train_dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=True
)

eval_dataset = CamouflageDataset(
    IMG_DIR,
    MASK_DIR,
    size=IMG_SIZE,
    augment=False
)

train_set = torch.utils.data.Subset(train_dataset, train_indices)
val_set = torch.utils.data.Subset(eval_dataset, val_indices)
test_set = torch.utils.data.Subset(eval_dataset, test_indices)

loader_generator = torch.Generator()
loader_generator.manual_seed(TRAIN_SEED)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY,
    generator=loader_generator
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

print(
    f"Total: {n_total} | Train: {len(train_set)} | "
    f"Val: {len(val_set)} | Test: {len(test_set)}"
)

model = SINet_GRA(pretrained=True).to(device)

bce_logits = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


def dice_score(pred, target, eps=1e-7):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def iou_score(pred, target, eps=1e-7):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def dice_loss(probs, targets, eps=1e-7):
    probs = probs.contiguous().view(probs.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1 - (2 * intersection + eps) / (union + eps)).mean()


train_losses = []
val_losses = []
train_dices = []
val_dices = []

best_val_dice = -1.0
best_val_iou = -1.0
best_val_loss = float("inf")
best_epoch = -1

torch.cuda.empty_cache()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss = 0.0
    running_train_dice = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        out = model(imgs)
        bce = bce_logits(out, masks)
        probs = torch.sigmoid(out)
        d_loss = dice_loss(probs, masks)
        loss = bce + d_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds_bin = (probs > 0.5).float()
        running_train_loss += loss.item()
        running_train_dice += dice_score(preds_bin, masks)

    avg_train_loss = running_train_loss / max(1, len(train_loader))
    avg_train_dice = running_train_dice / max(1, len(train_loader))

    train_losses.append(avg_train_loss)
    train_dices.append(avg_train_dice)

    model.eval()
    running_val_loss = 0.0
    running_val_dice = 0.0
    running_val_iou = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            out = model(imgs)
            bce = bce_logits(out, masks)
            probs = torch.sigmoid(out)
            d_loss = dice_loss(probs, masks)
            val_loss = bce + d_loss
            preds_bin = (probs > 0.5).float()

            running_val_loss += val_loss.item()
            running_val_dice += dice_score(preds_bin, masks)
            running_val_iou += iou_score(preds_bin, masks)

    avg_val_loss = running_val_loss / max(1, len(val_loader))
    avg_val_dice = running_val_dice / max(1, len(val_loader))
    avg_val_iou = running_val_iou / max(1, len(val_loader))

    scheduler.step(avg_val_dice)

    val_losses.append(avg_val_loss)
    val_dices.append(avg_val_dice)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Dice: {avg_train_dice:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Dice: {avg_val_dice:.4f} | "
        f"Val IoU: {avg_val_iou:.4f}"
    )

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_val_iou = avg_val_iou
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(" Saved best model:", BEST_MODEL_PATH)

print("\n========== BEST VALIDATION RESULT ==========")
print(f"Best epoch    : {best_epoch}")
print(f"Best Val Loss : {best_val_loss:.4f}")
print(f"Best Val Dice : {best_val_dice:.4f}")
print(f"Best Val IoU  : {best_val_iou:.4f}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"SINet + GRA Loss Curve (Train Seed {TRAIN_SEED})")
plt.tight_layout()
loss_curve_path = os.path.join(
    SAVE_DIR,
    f"sinet_gra_loss_curve_seed{TRAIN_SEED}.png"
)
plt.savefig(loss_curve_path)
plt.close()
print("Saved loss curve:", loss_curve_path)

plt.figure()
plt.plot(train_dices, label="Train Dice")
plt.plot(val_dices, label="Validation Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.title(f"SINet + GRA Dice Curve (Train Seed {TRAIN_SEED})")
plt.tight_layout()
dice_curve_path = os.path.join(
    SAVE_DIR,
    f"sinet_gra_dice_curve_seed{TRAIN_SEED}.png"
)
plt.savefig(dice_curve_path)
plt.close()
print("Saved Dice curve:", dice_curve_path)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)
model.eval()

test_loss_total = 0.0
test_dice_total = 0.0
test_iou_total = 0.0
test_sample_count = 0

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        out = model(imgs)
        bce = bce_logits(out, masks)
        probs = torch.sigmoid(out)
        d_loss = dice_loss(probs, masks)
        test_loss = bce + d_loss
        preds_bin = (probs > 0.5).float()
        current_batch_size = imgs.size(0)

        test_loss_total += test_loss.item() * current_batch_size
        test_dice_total += dice_score(preds_bin, masks) * current_batch_size
        test_iou_total += iou_score(preds_bin, masks) * current_batch_size
        test_sample_count += current_batch_size

avg_test_loss = test_loss_total / test_sample_count
avg_test_dice = test_dice_total / test_sample_count
avg_test_iou = test_iou_total / test_sample_count

print("\n========== FINAL TEST RESULTS ==========")
print(f"Training seed: {TRAIN_SEED}")
print(f"Test Loss    : {avg_test_loss:.4f}")
print(f"Test Dice    : {avg_test_dice:.4f}")
print(f"Test IoU     : {avg_test_iou:.4f}")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
num_samples_to_save = min(10, len(test_set))

for i in range(num_samples_to_save):
    img, mask = test_set[i]
    img_b = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_b)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()

    pred = (prob > 0.5).astype(np.uint8)
    gt = mask[0].numpy().astype(np.uint8)

    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    overlay = img_np.copy()
    overlay[gt == 1] = [255, 0, 0]
    overlay[pred == 1] = [0, 255, 0]

    prefix = os.path.join(
        PRED_DIR,
        f"seed{TRAIN_SEED}_sample_{i + 1}"
    )

    cv2.imwrite(prefix + "_image.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(prefix + "_gt.png", gt * 255)
    cv2.imwrite(prefix + "_pred.png", pred * 255)
    cv2.imwrite(prefix + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"Saved {num_samples_to_save} prediction samples in {PRED_DIR}/")