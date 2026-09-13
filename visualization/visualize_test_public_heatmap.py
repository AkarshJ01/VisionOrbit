import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DEFAULT_CHECKPOINT = os.path.join(str(PROJECT_ROOT), "checkpoints", "best_model_best_interval.pth")
CHECKPOINT = DEFAULT_CHECKPOINT if os.path.exists(DEFAULT_CHECKPOINT) else "checkpoints/best_model_best_interval.pth"

PATCH_SIZE = 256
THRESHOLD = 0.3

OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "outputs/visualizations/test_public_heatmaps")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# UNSEEN TEST CASES
# ============================================================

TEST_CASES = [
    {
        "aoi": "L15-0369E-1244N_1479_3214_13",
        "old_month": "2018_02",
        "new_month": "2019_12",
    },
    {
        "aoi": "L15-0509E-1108N_2037_3758_13",
        "old_month": "2018_02",
        "new_month": "2019_12",
    },
    {
        "aoi": "L15-1203E-1203N_4815_3379_13",
        "old_month": "2018_01",
        "new_month": "2019_09",
    },
]


# ============================================================
# FIND TIFF
# ============================================================

def find_image(aoi, month):

    folder = os.path.join(
        str(PROJECT_ROOT),
        "data",
        "test_public",
        aoi,
        "images_masked"
    )

    if not os.path.exists(folder):
        folder = os.path.join("data", "test_public", aoi, "images_masked")

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".tif") and month in f
    ]

    if not files:
        raise FileNotFoundError(
            f"No TIFF found for {aoi} {month}"
        )

    return os.path.join(folder, files[0])


# ============================================================
# LOAD MODEL
# ============================================================

print("Using device:", DEVICE)

model = UNet(in_channels=8, out_channels=1)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# RUN CNN ON FULL IMAGE
# ============================================================

def predict_full_image(old_img, new_img):

    height, width = old_img.shape[1:]

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # 4 x 4 = 16 patches for 1024x1024 image
    for row in range(0, height, PATCH_SIZE):

        for col in range(0, width, PATCH_SIZE):

            old_patch = old_img[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            new_patch = new_img[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            # ------------------------------------------------
            # [old 4 bands + new 4 bands] = 8 channels
            # ------------------------------------------------

            combined = np.concatenate(
                [old_patch, new_patch],
                axis=0
            )

            combined = combined.astype(
                np.float32
            ) / 255.0

            tensor = torch.from_numpy(
                combined
            ).unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # CNN prediction
            # ------------------------------------------------

            with torch.no_grad():

                logits = model(tensor)

                probability = torch.sigmoid(
                    logits
                )

            probability = probability[
                0, 0
            ].cpu().numpy()

            probability_map[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ] = probability

    return probability_map


# ============================================================
# PROCESS EACH UNSEEN AOI
# ============================================================

for case in TEST_CASES:

    aoi = case["aoi"]
    old_month = case["old_month"]
    new_month = case["new_month"]

    print("\n==========================================")
    print("AOI:", aoi)
    print("Period:", old_month, "->", new_month)
    print("==========================================")

    old_path = find_image(
        aoi,
        old_month
    )

    new_path = find_image(
        aoi,
        new_month
    )

    # --------------------------------------------------------
    # READ IMAGES
    # --------------------------------------------------------

    with rasterio.open(old_path) as src:

        old_img = src.read()

        transform = src.transform
        crs = src.crs

    with rasterio.open(new_path) as src:

        new_img = src.read()

    print("Image shape:", old_img.shape)
    print("CRS:", crs)

    # --------------------------------------------------------
    # RGB VISUALIZATION
    # --------------------------------------------------------

    old_rgb = np.transpose(
        old_img[:3],
        (1, 2, 0)
    ).astype(np.float32) / 255.0

    new_rgb = np.transpose(
        new_img[:3],
        (1, 2, 0)
    ).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # CNN INFERENCE
    # --------------------------------------------------------

    print("Running CNN...")

    probability_map = predict_full_image(
        old_img,
        new_img
    )

    # --------------------------------------------------------
    # BINARY CHANGE MASK
    # --------------------------------------------------------

    change_mask = (
        probability_map >= THRESHOLD
    )

    changed_pixels = int(
        change_mask.sum()
    )

    pixel_area = abs(
        transform.a * transform.e
    )

    changed_area = (
        changed_pixels * pixel_area
    )

    print(
        "Predicted changed pixels:",
        changed_pixels
    )

    print(
        "Predicted changed area:",
        f"{changed_area:.2f} m²"
    )

    print(
        "Maximum probability:",
        f"{probability_map.max():.4f}"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 6)
    )

    # --------------------------------------------------------
    # 1. BEFORE
    # --------------------------------------------------------

    axes[0].imshow(old_rgb)

    axes[0].set_title(
        f"BEFORE\n{old_month}",
        fontsize=14
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # 2. AFTER
    # --------------------------------------------------------

    axes[1].imshow(new_rgb)

    axes[1].set_title(
        f"AFTER\n{new_month}",
        fontsize=14
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # 3. CNN PROBABILITY HEATMAP
    # --------------------------------------------------------

    axes[2].imshow(
        probability_map,
        cmap="hot",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "CNN CHANGE PROBABILITY",
        fontsize=14
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # 4. FINAL BINARY CHANGE MASK
    # --------------------------------------------------------

    axes[3].imshow(
        new_rgb
    )

    axes[3].imshow(
        change_mask,
        alpha=0.45,
        interpolation="none"
    )

    axes[3].set_title(
        f"DETECTED CHANGE\n"
        f"Threshold = {THRESHOLD}",
        fontsize=14
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    fig.suptitle(
        f"SpaceNet 7 — UNSEEN TEST\n"
        f"{aoi}\n"
        f"{old_month} → {new_month}",
        fontsize=16
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    filename = (
        aoi +
        "_heatmap.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


print("\n==========================================")
print("DONE")
print("==========================================")
print(
    "Visualizations saved in:",
    OUTPUT_DIR
)