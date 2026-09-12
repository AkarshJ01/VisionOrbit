import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_long_interval import LongIntervalChangeDataset
from model import UNet


# =========================
# Configuration
# =========================

DATA_ROOT = "best_interval_patches"
CHECKPOINT = "checkpoints/best_model_best_interval.pth"

BATCH_SIZE = 4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using: {DEVICE}")


# =========================
# AOI split
# =========================

all_aois = sorted([
    name for name in __import__("os").listdir(DATA_ROOT)
    if __import__("os").path.isdir(
        __import__("os").path.join(DATA_ROOT, name)
    )
])

train_aois = all_aois[:8]
val_aois = all_aois[8:]

print(f"Training AOIs: {len(train_aois)}")
print(f"Validation AOIs: {len(val_aois)}")

print("Validation AOIs:")
for aoi in val_aois:
    print(f"  {aoi}")


# =========================
# Validation dataset
# =========================

val_dataset = LongIntervalChangeDataset(
    DATA_ROOT,
    val_aois,
    augment=False
)

print(f"Validation patches: {len(val_dataset)}")


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Model
# =========================

model = UNet(in_channels=8).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded.")


# =========================
# Collect predictions
# =========================

all_probs = []
all_masks = []

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_masks.append(
            masks.numpy()
        )


probs = np.concatenate(all_probs, axis=0)
masks = np.concatenate(all_masks, axis=0)

print(f"Prediction shape: {probs.shape}")
print(f"Mask shape: {masks.shape}")


# =========================
# Threshold sweep
# =========================

thresholds = np.arange(0.1, 1.0, 0.1)

print()
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<14}"
    f"{'Recall':<14}"
    f"{'Dice/F1':<14}"
    f"{'IoU':<14}"
)

print("=" * 70)


best_iou = -1
best_threshold = None


for threshold in thresholds:

    predictions = (
        probs >= threshold
    ).astype(np.uint8)

    ground_truth = (
        masks >= 0.5
    ).astype(np.uint8)


    tp = np.logical_and(
        predictions == 1,
        ground_truth == 1
    ).sum()

    fp = np.logical_and(
        predictions == 1,
        ground_truth == 0
    ).sum()

    fn = np.logical_and(
        predictions == 0,
        ground_truth == 1
    ).sum()


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )


    print(
        f"{threshold:<12.1f}"
        f"{precision:<14.4f}"
        f"{recall:<14.4f}"
        f"{dice:<14.4f}"
        f"{iou:<14.4f}"
    )


    if iou > best_iou:

        best_iou = iou
        best_threshold = threshold


# =========================
# Diagnostics
# =========================

ground_truth = (
    masks >= 0.5
).astype(np.uint8)

best_predictions = (
    probs >= best_threshold
).astype(np.uint8)


actual_pixels = ground_truth.sum()
predicted_pixels = best_predictions.sum()


print()
print("=" * 70)

print(
    f"BEST THRESHOLD: {best_threshold:.1f}"
)

print(
    f"BEST IoU:       {best_iou:.4f}"
)

print("=" * 70)


print()
print("Diagnostics:")

print(
    f"Actual changed pixels:    {actual_pixels}"
)

print(
    f"Predicted changed pixels: {predicted_pixels}"
)

if actual_pixels > 0:

    print(
        f"Predicted/actual ratio:   "
        f"{predicted_pixels / actual_pixels:.2f}x"
    )

print(
    f"Maximum probability:      "
    f"{probs.max():.4f}"
)

print(
    f"Mean probability:         "
    f"{probs.mean():.6f}"
)