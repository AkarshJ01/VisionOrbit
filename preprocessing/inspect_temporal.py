import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ROOT = os.path.join(str(PROJECT_ROOT), "data/patches/temporal")
ROOT = DEFAULT_ROOT if os.path.exists(DEFAULT_ROOT) else "temporal_patches"

# Pick a few AOIs/pairs to inspect
examples = [
    (
        "L15-0331E-1257N_1327_3160_13",
        "2018_06_to_2018_07"
    ),
    (
        "L15-0357E-1223N_1429_3296_13",
        "2018_06_to_2018_07"
    ),
    (
        "L15-0361E-1300N_1446_2989_13",
        "2019_06_to_2019_07"
    ),
    (
        "L15-0487E-1246N_1950_3207_13",
        "2019_11_to_2019_12"
    ),
]

fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)))

for row, (aoi, pair) in enumerate(examples):

    folder = os.path.join(ROOT, aoi, pair)

    if not os.path.exists(folder):
        print("Missing:", folder)
        continue

    # Find first patch that actually contains change
    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith("_mask.npy")
    )

    selected = None

    for mask_file in files:
        mask = np.load(os.path.join(folder, mask_file))

        if mask.sum() > 0:
            selected = mask_file.replace("_mask.npy", "")
            break

    if selected is None:
        print("No changed patch found:", aoi, pair)
        continue

    old = np.load(os.path.join(folder, selected + "_old.npy"))
    new = np.load(os.path.join(folder, selected + "_new.npy"))
    mask = np.load(os.path.join(folder, selected + "_mask.npy"))

    # Use first 3 bands as RGB
    old_rgb = np.transpose(old[:3], (1, 2, 0))
    new_rgb = np.transpose(new[:3], (1, 2, 0))

    # Normalize independently for visualization
    def normalize(img):
        img = img.astype(np.float32)

        lo = np.percentile(img, 2)
        hi = np.percentile(img, 98)

        img = (img - lo) / (hi - lo + 1e-8)

        return np.clip(img, 0, 1)

    old_rgb = normalize(old_rgb)
    new_rgb = normalize(new_rgb)

    # Old image
    axes[row, 0].imshow(old_rgb)
    axes[row, 0].set_title(f"{aoi}\nOld: {pair[:7]}")
    axes[row, 0].axis("off")

    # New image
    axes[row, 1].imshow(new_rgb)
    axes[row, 1].set_title(f"New: {pair[9:]}")
    axes[row, 1].axis("off")

    # Ground truth mask
    axes[row, 2].imshow(mask, cmap="gray")
    axes[row, 2].set_title(
        f"Ground truth\nChanged pixels: {int(mask.sum())}"
    )
    axes[row, 2].axis("off")

    # Overlay
    axes[row, 3].imshow(new_rgb)
    axes[row, 3].imshow(
        mask,
        cmap="Reds",
        alpha=0.45
    )
    axes[row, 3].set_title("New image + change mask")
    axes[row, 3].axis("off")

plt.tight_layout()

out_dir = os.path.join(str(PROJECT_ROOT), "outputs/visualizations")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "temporal_ground_truth_inspection.png")

plt.savefig(
    out_file,
    dpi=150,
    bbox_inches="tight"
)

print(f"\nSaved: {out_file}")