import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from dataset_loader import CamouflageDataset
from group_split import create_group_split
from models.sinet_se import SINet_SE


# -----------------------
# Config
# -----------------------
IMG_SIZE = 256
BATCH_SIZE = 2
pin_memory = True
EPOCHS = 20
LR = 1e-4

# -----------------------
# Seeds
# -----------------------
SPLIT_SEED = 42       # Keep fixed
TRAIN_SEED = 123      # Change this for the second run

SAVE_DIR = "checkpoints"
PRED_DIR = "pred_samples_sinet_se"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# -----------------------
# Reproducibility
# -----------------------
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

IMG_DIR = "mc_dataset_cropped/images"
MASK_DIR = "mc_dataset_cropped/masks"


# -----------------------
# Dataset + Fixed Split
# -----------------------
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

train_set = torch.utils.data.Subset(
    train_dataset,
    train_indices
)

val_set = torch.utils.data.Subset(
    eval_dataset,
    val_indices
)

test_set = torch.utils.data.Subset(
    eval_dataset,
    test_indices
)


# -----------------------
# DataLoaders
# -----------------------
loader_generator = torch.Generator()
loader_generator.manual_seed(TRAIN_SEED)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory,
    generator=loader_generator
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

print(
    f"Total: {n_total} | "
    f"Train: {len(train_set)} | "
    f"Val: {len(val_set)} | "
    f"Test: {len(test_set)}"
)


# -----------------------
# Model
# -----------------------
model = SINet_SE(pretrained=True).to(device)

bce_logits = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []
val_dices = []

best_val_dice = -1.0
best_val_iou = -1.0
best_epoch = -1


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
# Training loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / max(1, len(train_loader))
    train_losses.append(avg_train_loss)

    # -----------------------
    # Validation
    # -----------------------
    model.eval()

    running_val_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            out = model(imgs)

            bce = bce_logits(out, masks)
            probs = torch.sigmoid(out)
            d_loss = dice_loss(probs, masks)

            val_loss = bce + d_loss
            running_val_loss += val_loss.item()

            preds_bin = (probs > 0.5).float()

            running_dice += dice_score(
                preds_bin,
                masks
            )

            running_iou += iou_score(
                preds_bin,
                masks
            )

    avg_val_loss = running_val_loss / max(
        1,
        len(val_loader)
    )

    avg_val_dice = running_dice / max(
        1,
        len(val_loader)
    )

    avg_val_iou = running_iou / max(
        1,
        len(val_loader)
    )

    val_losses.append(avg_val_loss)
    val_dices.append(avg_val_dice)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Dice: {avg_val_dice:.4f} | "
        f"Val IoU: {avg_val_iou:.4f}"
    )

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_val_iou = avg_val_iou
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_DIR,
                "sinet_se_best_seed123.pth"
            )
        )

        print(
            " Saved best model:",
            os.path.join(
                SAVE_DIR,
                "sinet_se_best_seed123.pth"
            )
        )


print("\n========== BEST VALIDATION RESULT ==========")
print(f"Training seed : {TRAIN_SEED}")
print(f"Best epoch    : {best_epoch}")
print(f"Best Val Dice : {best_val_dice:.4f}")
print(f"Val IoU       : {best_val_iou:.4f}")


# -----------------------
# Save loss curve
# -----------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(
    f"SINet + SE Training Curve "
    f"(Train Seed {TRAIN_SEED})"
)
plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        f"sinet_se_loss_curve_seed{TRAIN_SEED}.png"
    )
)

plt.close()


# -----------------------
# Save prediction samples
# -----------------------
model.load_state_dict(
    torch.load(
        os.path.join(
            SAVE_DIR,
            "sinet_se_best_seed123.pth"
        ),
        map_location=device,
        weights_only=True
    )
)

model.eval()

for i in range(min(3, len(val_set))):
    img, mask = val_set[i]
    img_b = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_b)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()

    pred = (prob > 0.5).astype(np.uint8)
    gt = mask[0].numpy().astype(np.uint8)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    overlay = img_np.copy()
    overlay[gt == 1] = [255, 0, 0]
    overlay[pred == 1] = [0, 255, 0]

    cv2.imwrite(
        f"{PRED_DIR}/seed{TRAIN_SEED}_sample_{i+1}_image.png",
        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        f"{PRED_DIR}/seed{TRAIN_SEED}_sample_{i+1}_gt.png",
        gt * 255
    )

    cv2.imwrite(
        f"{PRED_DIR}/seed{TRAIN_SEED}_sample_{i+1}_pred.png",
        pred * 255
    )

    cv2.imwrite(
        f"{PRED_DIR}/seed{TRAIN_SEED}_sample_{i+1}_overlay.png",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

print(
    f"Saved prediction samples in {PRED_DIR}/"
)