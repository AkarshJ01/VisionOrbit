import glob
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.spacenet_dataset import SpaceNetChangeDataset
from models.unet import UNet


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using:", device)


# Validation dataset
PATCH_DIR = os.path.join(str(PROJECT_ROOT), "data/patches/base") if os.path.exists(os.path.join(str(PROJECT_ROOT), "data/patches/base")) else "patches"
dataset = SpaceNetChangeDataset(
    patches_dir=PATCH_DIR,
    split="val",
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)


# 12-channel model
model = UNet(
    in_channels=12,
    out_channels=1
).to(device)

CHECKPOINT = os.path.join(str(PROJECT_ROOT), "checkpoints", "best_model_12ch.pth")
if os.path.exists(CHECKPOINT):
    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device
        )
    )

model.eval()


# Collect predictions and ground truth
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


probabilities = np.concatenate(all_probabilities).reshape(-1)
masks = np.concatenate(all_masks).reshape(-1)


thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]


print()
print("=" * 70)
print("12-CHANNEL CNN THRESHOLD EVALUATION")
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


    tp = np.sum(
        (predictions == 1) &
        (ground_truth == 1)
    )

    fp = np.sum(
        (predictions == 1) &
        (ground_truth == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (ground_truth == 1)
    )


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
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{dice:<15.4f}"
        f"{iou:<15.4f}"
    )


best = max(
    results,
    key=lambda x: x[4]
)


print()
print("=" * 70)

print(
    f"BEST THRESHOLD: {best[0]:.2f}"
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

print()
print("Previous CNN baseline IoU: 0.1651")
print(f"12-channel CNN IoU:        {best[4]:.4f}")
