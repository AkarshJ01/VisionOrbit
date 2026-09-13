import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.spacenet_dataset import SpaceNetChangeDataset
from models.unet import UNet


def dice_score(pred, target, threshold=0.5):

    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum()

    dice = (
        (2 * intersection + 1e-8)
        / (pred.sum() + target.sum() + 1e-8)
    )

    return dice.item()


def iou_score(pred, target, threshold=0.5):

    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    iou = (intersection + 1e-8) / (union + 1e-8)

    return iou.item()


class DiceLoss(nn.Module):

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()

        dice = (
            (2 * intersection + 1e-8)
            / (probs.sum() + targets.sum() + 1e-8)
        )

        return 1 - dice


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using:", device)


PATCHES_DIR = os.path.join(str(PROJECT_ROOT), "data/patches/base") if os.path.exists(os.path.join(str(PROJECT_ROOT), "data/patches/base")) else "patches"

# Dataset
train_dataset = SpaceNetChangeDataset(
    patches_dir=PATCHES_DIR,
    split="train",
    augment=False
)

val_dataset = SpaceNetChangeDataset(
    patches_dir=PATCHES_DIR,
    split="val",
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


# Model
model = UNet(
    in_channels=12,
    out_channels=1
).to(device)


print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# Loss
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()


def combined_loss(pred, target):

    return (
        bce_loss(pred, target)
        + dice_loss(pred, target)
    )


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


num_epochs = 25

best_iou = 0.0


for epoch in range(num_epochs):

    # =========================
    # TRAIN
    # =========================

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                masks
            )

            val_iou += iou_score(
                outputs,
                masks
            )

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)


    print(
        f"Epoch {epoch+1:02d}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f}"
    )


    # Save best model
    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(
            model.state_dict(),
            "checkpoints/best_model_12ch.pth"
        )

        print(
            f"  Saved new best model! IoU: {best_iou:.4f}"
        )


print("\nTraining complete.")
print(f"Best validation IoU: {best_iou:.4f}")
