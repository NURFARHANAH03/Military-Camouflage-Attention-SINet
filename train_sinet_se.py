import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import cv2

from dataset_loader import CamouflageDataset
from models.sinet_se import SINet_SE

# -----------------------
# Config (Dell G7 GTX 1060 6GB)
# -----------------------
IMG_SIZE = 256
BATCH_SIZE = 2
pin_memory = True
EPOCHS = 20
LR = 1e-4

SAVE_DIR = "checkpoints"
PRED_DIR = "pred_samples_sinet_se"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMG_DIR = "mc_dataset_cropped/images"
MASK_DIR = "mc_dataset_cropped/masks"

# -----------------------
# Dataset + Split
# -----------------------
full_dataset = CamouflageDataset(IMG_DIR, MASK_DIR, size=IMG_SIZE, augment=False)

n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_idx, val_idx, test_idx = random_split(
    range(n_total),
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = CamouflageDataset(IMG_DIR, MASK_DIR, size=IMG_SIZE, augment=True)
val_dataset = CamouflageDataset(IMG_DIR, MASK_DIR, size=IMG_SIZE, augment=False)

train_set = torch.utils.data.Subset(train_dataset, train_idx.indices)
val_set = torch.utils.data.Subset(val_dataset, val_idx.indices)
test_set = torch.utils.data.Subset(val_dataset, test_idx.indices)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

print(f"Total: {n_total} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# -----------------------
# Model
# -----------------------
model = SINet_SE(pretrained=True).to(device)

bce_logits = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses, val_dices = [], [], []
best_val_dice = -1


def dice_score(pred, target, eps=1e-7):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_score(pred, target, eps=1e-7):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def dice_loss(probs, targets, eps=1e-7):
    probs = probs.contiguous().view(probs.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()


torch.cuda.empty_cache()

# -----------------------
# Train loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        masks = masks.float()

        out = model(imgs)

        bce = bce_logits(out, masks)
        probs = torch.sigmoid(out)
        d = dice_loss(probs, masks)

        loss = bce + d

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_train = running_loss / max(1, len(train_loader))
    train_losses.append(avg_train)

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    running_vloss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            masks = masks.float()

            out = model(imgs)

            bce = bce_logits(out, masks)
            probs = torch.sigmoid(out)
            d = dice_loss(probs, masks)

            vloss = bce + d
            running_vloss += vloss.item()

            preds_bin = (probs > 0.5).float()
            running_dice += dice_score(preds_bin, masks)
            running_iou += iou_score(preds_bin, masks)

    avg_val = running_vloss / max(1, len(val_loader))
    avg_dice = running_dice / max(1, len(val_loader))
    avg_iou = running_iou / max(1, len(val_loader))

    val_losses.append(avg_val)
    val_dices.append(avg_dice)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {avg_train:.4f} | "
        f"Val Loss: {avg_val:.4f} | "
        f"Val Dice: {avg_dice:.4f} | "
        f"Val IoU: {avg_iou:.4f}"
    )

    if avg_dice > best_val_dice:
        best_val_dice = avg_dice
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "sinet_se_best.pth"))
        print(" Saved best model (by Dice):", os.path.join(SAVE_DIR, "sinet_se_best.pth"))

# -----------------------
# Save loss curve
# -----------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("SINet + SE Training Curve")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sinet_se_loss_curve.png"))
print(" Saved loss curve:", os.path.join(SAVE_DIR, "sinet_se_loss_curve.png"))

# -----------------------
# Save 3 prediction samples
# -----------------------
model.load_state_dict(
    torch.load(
        os.path.join(SAVE_DIR, "sinet_se_best.pth"),
        map_location=device,
        weights_only=True
    )
)
model.eval()

for i in range(3):
    img, mask = val_set[i]
    img_b = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_b)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()

    pred = (prob > 0.5).astype(np.uint8)
    gt = mask[0].numpy().astype(np.uint8)

    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    overlay = img_np.copy()
    overlay[gt == 1] = [255, 0, 0]
    overlay[pred == 1] = [0, 255, 0]

    cv2.imwrite(f"{PRED_DIR}/sample_{i+1}_image.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{PRED_DIR}/sample_{i+1}_gt.png", gt * 255)
    cv2.imwrite(f"{PRED_DIR}/sample_{i+1}_pred.png", pred * 255)
    cv2.imwrite(f"{PRED_DIR}/sample_{i+1}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f" Saved 3 prediction samples in {PRED_DIR}/")