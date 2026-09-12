import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_long_interval import LongIntervalChangeDataset
from model import UNet


# ============================================================
# CONFIG
# ============================================================

ROOT = "long_interval_patches"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 25


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)


# ============================================================
# AOI SPLIT
# ============================================================

aois = sorted(
    [
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d))
    ]
)

train_aois = aois[:8]
val_aois = aois[8:]

print("Training AOIs:", len(train_aois))
print("Validation AOIs:", len(val_aois))


# ============================================================
# DATASETS
# ============================================================

train_dataset = LongIntervalChangeDataset(
    ROOT,
    train_aois,
    augment=False
)

val_dataset = LongIntervalChangeDataset(
    ROOT,
    val_aois,
    augment=False
)

print("Training patches:", len(train_dataset))
print("Validation patches:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = UNet(in_channels=8).to(device)

print("Model parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1.0

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)

    dice = (
        (2.0 * intersection + smooth)
        /
        (
            pred_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + smooth
        )
    )

    return 1.0 - dice.mean()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best_model_long_interval.pth"
)

best_iou = 0.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_bce = bce_loss(outputs, masks)
        loss_dice = dice_loss(outputs, masks)

        loss = loss_bce + loss_dice

        loss.backward()

        optimizer.step()

        train_loss_total += loss.item()

    train_loss = train_loss_total / len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)

            loss = loss_bce + loss_dice

            val_loss_total += loss.item()

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Binary prediction
            preds = (probs > 0.5).float()

            # Flatten each image
            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)

            intersection = (
                preds_flat * masks_flat
            ).sum(dim=1)

            pred_area = preds_flat.sum(dim=1)
            mask_area = masks_flat.sum(dim=1)

            union = (
                pred_area
                + mask_area
                - intersection
            )

            dice = (
                (2 * intersection + 1e-7)
                /
                (pred_area + mask_area + 1e-7)
            )

            iou = (
                (intersection + 1e-7)
                /
                (union + 1e-7)
            )

            dice_total += dice.mean().item()
            iou_total += iou.mean().item()

    val_loss = val_loss_total / len(val_loader)
    val_dice = dice_total / len(val_loader)
    val_iou = iou_total / len(val_loader)


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"  ✓ Saved best model "
            f"(IoU: {best_iou:.4f})"
        )


# ============================================================
# DONE
# ============================================================

print("\nTraining complete.")
print(f"Best validation IoU: {best_iou:.4f}")
print(f"Model saved to: {checkpoint_path}")
