import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_long_interval import LongIntervalChangeDataset
from model import UNet


# ============================================================
# CONFIG
# ============================================================

ROOT = "long_interval_patches"
CHECKPOINT = "checkpoints/best_model_long_interval.pth"

BATCH_SIZE = 4

THRESHOLDS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


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

print("Validation AOIs:")
for aoi in val_aois:
    print(" ", aoi)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = LongIntervalChangeDataset(
    ROOT,
    val_aois,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation patches:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = UNet(in_channels=8).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print("Loaded:", CHECKPOINT)


# ============================================================
# COLLECT ALL PREDICTIONS
# ============================================================

all_probs = []
all_masks = []

print("\nRunning inference...")

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_masks.append(
            masks.numpy()
        )


# Concatenate all validation batches
all_probs = np.concatenate(all_probs, axis=0)
all_masks = np.concatenate(all_masks, axis=0)

print("Predictions shape:", all_probs.shape)
print("Masks shape:", all_masks.shape)


# ============================================================
# EVALUATE EACH THRESHOLD GLOBALLY
# ============================================================

print("\n")
print("=" * 75)
print("GLOBAL VALIDATION RESULTS")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Dice/F1':<12}"
    f"{'IoU':<12}"
)

print("-" * 75)


best_iou = -1
best_threshold = None

for threshold in THRESHOLDS:

    # Binary predictions
    preds = all_probs >= threshold

    # Ground truth
    targets = all_masks >= 0.5

    # Global pixel counts
    tp = np.logical_and(
        preds,
        targets
    ).sum()

    fp = np.logical_and(
        preds,
        np.logical_not(targets)
    ).sum()

    fn = np.logical_and(
        np.logical_not(preds),
        targets
    ).sum()

    # Metrics
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    print(
        f"{threshold:<12.1f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{dice:<12.4f}"
        f"{iou:<12.4f}"
    )

    if iou > best_iou:

        best_iou = iou
        best_threshold = threshold


# ============================================================
# BEST RESULT
# ============================================================

print("\n" + "=" * 75)

print(
    f"Best global IoU: {best_iou:.4f}"
)

print(
    f"Best threshold: {best_threshold:.1f}"
)

print("=" * 75)


# ============================================================
# EXTRA DIAGNOSTICS
# ============================================================

best_preds = all_probs >= best_threshold
targets = all_masks >= 0.5

predicted_pixels = best_preds.sum()
actual_pixels = targets.sum()

print("\nDiagnostics:")
print("Actual changed pixels:", int(actual_pixels))
print("Predicted changed pixels:", int(predicted_pixels))

print(
    "Predicted/actual ratio:",
    f"{predicted_pixels / actual_pixels:.2f}x"
    if actual_pixels > 0
    else "N/A"
)

print(
    "Maximum probability:",
    f"{all_probs.max():.4f}"
)

print(
    "Mean probability:",
    f"{all_probs.mean():.6f}"
)
