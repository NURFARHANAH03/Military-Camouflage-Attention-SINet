import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import cv2

from dataset_loader import CamouflageDataset

# When using Unet without pretrained
#from models.unet import UNet  # make sure this path is correct

# When using Unet with pretrained RESNET 50
from models.unet_pretrained import UNetResNet
model = UNetResNet()

# -----------------------
# Config (Dell G7 GTX 1060 6GB)
# -----------------------
IMG_SIZE = 256
BATCH_SIZE = 2
pin_memory = True
EPOCHS = 20
LR = 1e-4
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Dataset + Split
# -----------------------

# full dataset for counting only
full_dataset = CamouflageDataset(
    "mc_dataset_cropped/images",
    "mc_dataset_cropped/masks",
    size=IMG_SIZE,
    augment=False
)

n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

# split indices only
train_idx, val_idx, test_idx = random_split(
    range(n_total),
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

# training dataset = augmentation ON
train_dataset = CamouflageDataset(
    "mc_dataset_cropped/images",
    "mc_dataset_cropped/masks",
    size=IMG_SIZE,
    augment=True
)

# validation/test dataset = augmentation OFF
val_dataset = CamouflageDataset(
    "mc_dataset_cropped/images",
    "mc_dataset_cropped/masks",
    size=IMG_SIZE,
    augment=False
)

# apply same indices
train_set = torch.utils.data.Subset(train_dataset, train_idx.indices)
val_set   = torch.utils.data.Subset(val_dataset, val_idx.indices)
test_set  = torch.utils.data.Subset(val_dataset, test_idx.indices)

# loaders
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
#Unet without pretrained
#model = UNet().to(device)

#Unet with pretrained RESNET 50
model = UNetResNet().to(device)

# If your UNet returns logits (not sigmoid), use BCEWithLogitsLoss.
# If your UNet already applies sigmoid, use BCELoss.
# We'll handle both by checking output range quickly.
bce_logits = nn.BCEWithLogitsLoss()

def dice_score(pred, target, eps=1e-7):
    # pred/target shape: [B,1,H,W] with 0/1
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

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses, val_dices = [], [], []
best_val_dice = -1

def dice_loss(probs, targets, eps=1e-7):
    probs = probs.contiguous().view(probs.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()

# -----------------------
# Train loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        out = model(imgs)  # [B,1,H,W] either logits or prob
        # convert to logits-loss safe: if out already sigmoid, clamp & use BCELoss instead
        if out.min().item() >= 0.0 and out.max().item() <= 1.0:
            probs = out
            bce = nn.BCELoss()(probs, masks)
        else:
            probs = torch.sigmoid(out)
            bce = bce_logits(out, masks)

        d = dice_loss(probs, masks)
        loss = bce + d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train = running_loss / max(1, len(train_loader))
    train_losses.append(avg_train)

    # Validation
    model.eval()
    running_vloss = 0.0
    running_dice = 0.0
    running_iou = 0.0


    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)

            if out.min().item() >= 0.0 and out.max().item() <= 1.0:
                probs = out
                bce = nn.BCELoss()(probs, masks)
            else:
                probs = torch.sigmoid(out)
                bce = bce_logits(out, masks)

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

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}")

    # Save best model based on Dice
    if avg_dice > best_val_dice:
        best_val_dice = avg_dice
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "unet_best.pth"))
        print(" Saved best model (by Dice):", os.path.join(SAVE_DIR, "unet_best.pth"))

# Save a quick loss curve for slides
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("U-Net Training Curve")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "unet_loss_curve.png"))
print(" Saved loss curve:", os.path.join(SAVE_DIR, "unet_loss_curve.png"))

# -----------------------
# Save 3 prediction samples (for slides)
# -----------------------
os.makedirs("pred_samples", exist_ok=True)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "unet_best.pth"), map_location=device))
model.eval()

# take 3 samples from val_set
for i in range(3):
    img, mask = val_set[i]
    img_b = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_b)
        if out.min().item() >= 0.0 and out.max().item() <= 1.0:
            prob = out[0,0].cpu().numpy()
        else:
            prob = torch.sigmoid(out)[0,0].cpu().numpy()

    pred = (prob > 0.3).astype(np.uint8)
    gt = mask[0].numpy().astype(np.uint8)

    # Convert original image tensor back to uint8 RGB
    img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)

    # Create overlay: green=pred, red=gt
    overlay = img_np.copy()
    overlay[gt == 1] = [255, 0, 0]      # GT red
    overlay[pred == 1] = [0, 255, 0]    # Pred green

    cv2.imwrite(f"pred_samples/sample_{i+1}_image.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"pred_samples/sample_{i+1}_gt.png", (gt * 255))
    cv2.imwrite(f"pred_samples/sample_{i+1}_pred.png", (pred * 255))
    cv2.imwrite(f"pred_samples/sample_{i+1}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(" Saved 3 prediction samples in pred_samples/")