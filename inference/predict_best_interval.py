import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.long_interval_dataset import LongIntervalChangeDataset
from models.unet import UNet


# =========================
# Configuration
# =========================

DEFAULT_DATA_ROOT = os.path.join(str(PROJECT_ROOT), "data/patches/best_interval")
DATA_ROOT = DEFAULT_DATA_ROOT if os.path.exists(DEFAULT_DATA_ROOT) else "best_interval_patches"

DEFAULT_CHECKPOINT = os.path.join(str(PROJECT_ROOT), "checkpoints", "best_model_best_interval.pth")
CHECKPOINT = DEFAULT_CHECKPOINT if os.path.exists(DEFAULT_CHECKPOINT) else "checkpoints/best_model_best_interval.pth"

THRESHOLD = 0.3

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using: {DEVICE}")


# =========================
# Validation AOIs
# =========================

all_aois = sorted([
    name for name in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, name))
])

train_aois = all_aois[:8]
val_aois = all_aois[8:]

print(f"Validation AOIs: {val_aois}")


# =========================
# Dataset
# =========================

dataset = LongIntervalChangeDataset(
    DATA_ROOT,
    val_aois,
    augment=False
)

print(f"Validation patches: {len(dataset)}")


# =========================
# Find a patch containing change
# =========================

selected_index = None

for i in range(len(dataset)):

    image, mask = dataset[i]

    if mask.sum() > 0:

        selected_index = i

        print(
            f"Selected validation patch: {i}"
        )

        print(
            f"Ground truth changed pixels: "
            f"{int(mask.sum().item())}"
        )

        break


if selected_index is None:

    print("No validation patch contains change.")
    exit()


# =========================
# Load model
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
# Run prediction
# =========================

image, mask = dataset[selected_index]

input_tensor = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    output = model(input_tensor)

    probability = torch.sigmoid(output)[0, 0].cpu().numpy()


prediction = (
    probability >= THRESHOLD
).astype(np.uint8)


# =========================
# Prepare images
# =========================

image_np = image.numpy()

old_image = image_np[:4]
new_image = image_np[4:8]

old_rgb = np.transpose(
    old_image[:3],
    (1, 2, 0)
)

new_rgb = np.transpose(
    new_image[:3],
    (1, 2, 0)
)

ground_truth = mask[0].numpy()


# =========================
# Statistics
# =========================

print()
print("=" * 50)

print(
    f"Ground truth changed pixels: "
    f"{ground_truth.sum():.0f}"
)

print(
    f"Predicted changed pixels:     "
    f"{prediction.sum():.0f}"
)

print(
    f"Maximum probability:           "
    f"{probability.max():.4f}"
)

print(
    f"Mean probability:              "
    f"{probability.mean():.6f}"
)

print("=" * 50)


# =========================
# Visualization
# =========================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 5)
)


axes[0].imshow(old_rgb)
axes[0].set_title("Before (2018)")
axes[0].axis("off")


axes[1].imshow(new_rgb)
axes[1].set_title("After (2019)")
axes[1].axis("off")


axes[2].imshow(ground_truth)
axes[2].set_title("Ground Truth")
axes[2].axis("off")


axes[3].imshow(prediction)
axes[3].set_title(
    f"CNN Prediction (threshold={THRESHOLD})"
)
axes[3].axis("off")


plt.tight_layout()

out_dir = os.path.join(str(PROJECT_ROOT), "outputs/predictions")
os.makedirs(out_dir, exist_ok=True)
OUTPUT = os.path.join(out_dir, "best_interval_prediction.png")

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print(f"Saved visualization to: {OUTPUT}")
