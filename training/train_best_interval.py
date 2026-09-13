import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.long_interval_dataset import LongIntervalChangeDataset
from models.unet import UNet


# ============================================================
# CONFIG
# ============================================================

DEFAULT_ROOT = os.path.join(str(PROJECT_ROOT), "data/patches/best_interval")
ROOT = DEFAULT_ROOT if os.path.exists(DEFAULT_ROOT) else "best_interval_patches"
CHECKPOINT_DIR = os.path.join(str(PROJECT_ROOT), "checkpoints")

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

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1.0

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (
        pred_flat * target_flat
    ).sum(dim=1)

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

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best_model_best_interval.pth"
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

        loss_bce = bce_loss(
            outputs,
            masks
        )

        loss_dice = dice_loss(
            outputs,
            masks
        )

        loss = loss_bce + loss_dice

        loss.backward()

        optimizer.step()

        train_loss_total += loss.item()

    train_loss = (
        train_loss_total
        / len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    # Accumulate global pixel counts
    tp_total = 0
    fp_total = 0
    fn_total = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = loss_bce + loss_dice

            val_loss_total += loss.item()

            # Threshold at 0.5 for training-time monitoring
            probs = torch.sigmoid(outputs)

            preds = probs > 0.5
            targets = masks > 0.5

            tp_total += torch.logical_and(
                preds,
                targets
            ).sum().item()

            fp_total += torch.logical_and(
                preds,
                torch.logical_not(targets)
            ).sum().item()

            fn_total += torch.logical_and(
                torch.logical_not(preds),
                targets
            ).sum().item()


    val_loss = (
        val_loss_total
        / len(val_loader)
    )

    # Global IoU
    denominator = (
        tp_total
        + fp_total
        + fn_total
    )

    if denominator > 0:
        val_iou = tp_total / denominator
    else:
        val_iou = 0.0

    # Global Dice
    dice_denominator = (
        2 * tp_total
        + fp_total
        + fn_total
    )

    if dice_denominator > 0:
        val_dice = (
            2 * tp_total
            / dice_denominator
        )
    else:
        val_dice = 0.0


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
    # SAVE BEST
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

print(
    f"Best validation IoU: {best_iou:.4f}"
)

print(
    f"Model saved to: {checkpoint_path}"
)
