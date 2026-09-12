import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpaceNetChangeDataset
from model import UNet


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Find validation patches
# --------------------------------------------------

all_patches = sorted(
    glob.glob("patches/*/*_old.npy")
)

aois = sorted(
    set(
        os.path.basename(os.path.dirname(p))
        for p in all_patches
    )
)

# Same split we used during training:
# first 8 AOIs = training
# last 2 AOIs = validation

val_aois = aois[8:]

val_files = [
    p for p in all_patches
    if os.path.basename(os.path.dirname(p)) in val_aois
]

dataset = SpaceNetChangeDataset(val_files)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)

print("Validation patches:", len(dataset))


# --------------------------------------------------
# Store predictions and ground truth
# --------------------------------------------------

all_probabilities = []
all_masks = []


with torch.no_grad():

    for images, masks in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_masks.append(
            masks.numpy()
        )


probabilities = np.concatenate(
    all_probabilities,
    axis=0
)

masks = np.concatenate(
    all_masks,
    axis=0
)


# --------------------------------------------------
# Evaluate different thresholds
# --------------------------------------------------

thresholds = [
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


print()
print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'Dice/F1':<15}"
    f"{'IoU':<15}"
)

print("-" * 70)


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    ground_truth = (
        masks >= 0.5
    ).astype(np.uint8)


    # Flatten everything
    pred = predictions.reshape(-1)
    true = ground_truth.reshape(-1)


    # Confusion matrix
    tp = np.sum(
        (pred == 1) & (true == 1)
    )

    fp = np.sum(
        (pred == 1) & (true == 0)
    )

    fn = np.sum(
        (pred == 0) & (true == 1)
    )


    # Metrics
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


    results.append(
        (threshold, precision, recall, dice, iou)
    )


    print(
        f"{threshold:<12.1f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{dice:<15.4f}"
        f"{iou:<15.4f}"
    )


# --------------------------------------------------
# Best threshold
# --------------------------------------------------

best = max(
    results,
    key=lambda x: x[4]
)

print()
print("=" * 70)

print(
    f"BEST THRESHOLD: {best[0]:.1f}"
)

print(
    f"Precision: {best[1]:.4f}"
)

print(
    f"Recall:    {best[2]:.4f}"
)

print(
    f"Dice/F1:   {best[3]:.4f}"
)

print(
    f"IoU:       {best[4]:.4f}"
)

print("=" * 70)
