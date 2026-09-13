import glob
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Find validation patches
# --------------------------------------------------

PATCH_DIR = os.path.join(str(PROJECT_ROOT), "data/patches/base") if os.path.exists(os.path.join(str(PROJECT_ROOT), "data/patches/base")) else "patches"
all_patches = sorted(
    glob.glob(os.path.join(PATCH_DIR, "*", "*_old.npy"))
)

aois = sorted(
    set(
        os.path.basename(
            os.path.dirname(p)
        )
        for p in all_patches
    )
)

# Same validation split as before
val_aois = aois[8:]

val_files = [
    p for p in all_patches
    if os.path.basename(
        os.path.dirname(p)
    ) in val_aois
]

print("Validation patches:", len(val_files))


# --------------------------------------------------
# Load all validation data
# --------------------------------------------------

all_differences = []
all_masks = []


for old_path in val_files:

    new_path = old_path.replace(
        "_old.npy",
        "_new.npy"
    )

    mask_path = old_path.replace(
        "_old.npy",
        "_mask.npy"
    )

    old = np.load(old_path).astype(
        np.float32
    ) / 255.0

    new = np.load(new_path).astype(
        np.float32
    ) / 255.0

    mask = np.load(mask_path).astype(
        np.uint8
    )

    # ------------------------------------------------
    # Calculate image difference
    # ------------------------------------------------
    #
    # old/new have shape:
    # (4, 256, 256)
    #
    # Average absolute difference
    # across the 4 bands.
    # ------------------------------------------------

    difference = np.abs(
        new - old
    ).mean(axis=0)

    all_differences.append(
        difference
    )

    all_masks.append(
        mask
    )


differences = np.concatenate(
    [x.reshape(-1) for x in all_differences]
)

masks = np.concatenate(
    [x.reshape(-1) for x in all_masks]
)


# --------------------------------------------------
# Evaluate thresholds
# --------------------------------------------------

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50
]


print()
print("=" * 70)
print("IMAGE DIFFERENCE BASELINE")
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
        differences >= threshold
    ).astype(np.uint8)

    ground_truth = (
        masks >= 0.5
    ).astype(np.uint8)


    tp = np.sum(
        (predictions == 1)
        & (ground_truth == 1)
    )

    fp = np.sum(
        (predictions == 1)
        & (ground_truth == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (ground_truth == 1)
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
        2 * tp
        / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0
    )

    iou = (
        tp
        / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )


    results.append(
        (
            threshold,
            precision,
            recall,
            dice,
            iou
        )
    )


    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{dice:<15.4f}"
        f"{iou:<15.4f}"
    )


# --------------------------------------------------
# Best result
# --------------------------------------------------

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
print("CNN baseline IoU: approximately 0.165")
print(
    f"Difference baseline IoU: {best[4]:.4f}"
)
