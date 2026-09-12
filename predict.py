import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio

from dataset import SpaceNetChangeDataset
from model import UNet


# ==================================================
# DEVICE
# ==================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)


# ==================================================
# LOAD ORIGINAL BEST MODEL
# ==================================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()


# ==================================================
# FIND VALIDATION PATCHES
# ==================================================

all_patches = sorted(
    glob.glob("patches/*/*_old.npy")
)

aois = sorted(
    set(
        os.path.basename(os.path.dirname(p))
        for p in all_patches
    )
)

val_aois = aois[8:]

val_files = [
    p for p in all_patches
    if os.path.basename(os.path.dirname(p)) in val_aois
]

dataset = SpaceNetChangeDataset(val_files)


# ==================================================
# FIND A PATCH THAT ACTUALLY HAS CHANGE
# ==================================================

chosen_index = None

for i in range(len(dataset)):

    _, mask = dataset[i]

    if mask.sum() > 0:
        chosen_index = i
        break

if chosen_index is None:
    chosen_index = 0


print("Using validation patch:", chosen_index)
print("File:", val_files[chosen_index])


# ==================================================
# LOAD PATCH
# ==================================================

image, mask = dataset[chosen_index]

input_tensor = image.unsqueeze(0).to(device)


# ==================================================
# PREDICT
# ==================================================

with torch.no_grad():

    prediction = model(input_tensor)

    probability = torch.sigmoid(prediction)[0, 0].cpu().numpy()

predicted_mask = (probability > 0.5).astype(np.uint8)

ground_truth = mask[0].numpy()


# ==================================================
# GET RGB FOR VISUALIZATION
# ==================================================

old_path = val_files[chosen_index]
new_path = old_path.replace("_old.npy", "_new.npy")

old = np.load(old_path).astype(np.float32) / 255.0
new = np.load(new_path).astype(np.float32) / 255.0

# Use first 3 bands as RGB
old_rgb = np.transpose(old[:3], (1, 2, 0))
new_rgb = np.transpose(new[:3], (1, 2, 0))


# ==================================================
# CREATE VISUALIZATION
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(old_rgb)
axes[0, 0].set_title("2018 Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(new_rgb)
axes[0, 1].set_title("2019 Image")
axes[0, 1].axis("off")

axes[1, 0].imshow(ground_truth)
axes[1, 0].set_title(
    f"Ground Truth ({int(ground_truth.sum())} changed pixels)"
)
axes[1, 0].axis("off")

axes[1, 1].imshow(predicted_mask)
axes[1, 1].set_title(
    f"CNN Prediction ({int(predicted_mask.sum())} pixels)"
)
axes[1, 1].axis("off")

plt.tight_layout()

output = "cnn_prediction.png"

plt.savefig(output, dpi=150)

print("\nSaved:", output)
print("Ground truth changed pixels:", int(ground_truth.sum()))
print("Predicted changed pixels:", int(predicted_mask.sum()))
print(
    "Maximum predicted probability:",
    round(float(probability.max()), 4)
)
