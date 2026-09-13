import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_RESULTS = os.path.join(str(PROJECT_ROOT), "outputs/results/test_public_results.json")
RESULTS_FILE = DEFAULT_RESULTS if os.path.exists(DEFAULT_RESULTS) else "test_public_results.json"

OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "outputs/visualizations/test_public_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)


# ============================================================
# HELPER — FIND IMAGE FILE
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
        # fallback
        folder = os.path.join("data", "test_public", aoi, "images_masked")

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    # SpaceNet filenames contain the month and AOI.
    candidates = [
        f for f in os.listdir(folder)
        if month in f and f.endswith(".tif")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find image for {aoi}, {month}"
        )

    return os.path.join(folder, candidates[0])


# ============================================================
# CREATE VISUALIZATION FOR EACH AOI
# ============================================================

for test in results["tests"]:

    aoi = test["aoi"]
    old_month = test["old_month"]
    new_month = test["new_month"]

    print("\nProcessing:", aoi)
    print(f"Period: {old_month} -> {new_month}")

    old_path = find_image(aoi, old_month)
    new_path = find_image(aoi, new_month)

    print("Before:", old_path)
    print("After :", new_path)

    # --------------------------------------------------------
    # READ IMAGES
    # --------------------------------------------------------

    with rasterio.open(old_path) as src:
        old = src.read()

    with rasterio.open(new_path) as src:
        new = src.read()

    print("Image shape:", old.shape)

    # --------------------------------------------------------
    # CONVERT 4-BAND IMAGE -> RGB
    # --------------------------------------------------------

    old_rgb = np.transpose(
        old[:3],
        (1, 2, 0)
    ).astype(np.float32) / 255.0

    new_rgb = np.transpose(
        new[:3],
        (1, 2, 0)
    ).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # CREATE DETECTION MASK
    # --------------------------------------------------------
    #
    # Your JSON contains the final detected regions.
    # We reconstruct a mask from their bounding boxes.
    #

    detection_mask = np.zeros(
        old_rgb.shape[:2],
        dtype=bool
    )

    for event in test["events"]:

        x1, y1, x2, y2 = event["bbox_pixels"]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(old_rgb.shape[1], int(x2))
        y2 = min(old_rgb.shape[0], int(y2))

        detection_mask[
            y1:y2,
            x1:x2
        ] = True

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # ========================================================
    # 1. BEFORE
    # ========================================================

    axes[0].imshow(old_rgb)

    axes[0].set_title(
        f"BEFORE\n{old_month}",
        fontsize=14
    )

    axes[0].axis("off")


    # ========================================================
    # 2. AFTER
    # ========================================================

    axes[1].imshow(new_rgb)

    axes[1].set_title(
        f"AFTER\n{new_month}",
        fontsize=14
    )

    axes[1].axis("off")


    # ========================================================
    # 3. DETECTED CHANGES
    # ========================================================

    axes[2].imshow(new_rgb)

    # Show detected regions as transparent overlay
    axes[2].imshow(
        np.ma.masked_where(
            ~detection_mask,
            detection_mask
        ),
        alpha=0.40,
        interpolation="none"
    )

    # Draw bounding boxes around final events
    for i, event in enumerate(test["events"]):

        x1, y1, x2, y2 = event["bbox_pixels"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2
        )

        axes[2].add_patch(rect)

        # Event number
        axes[2].text(
            x1,
            y1,
            str(i + 1),
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

    axes[2].set_title(
        f"DETECTED CHANGES\n"
        f"{len(test['events'])} events",
        fontsize=14
    )

    axes[2].axis("off")


    # ========================================================
    # OVERALL TITLE
    # ========================================================

    fig.suptitle(
        f"SpaceNet 7 — Unseen Test AOI\n"
        f"{aoi}\n"
        f"{old_month} → {new_month}",
        fontsize=15
    )

    plt.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================

    filename = (
        aoi.replace("/", "_")
        + "_results.png"
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

    print("Saved:", output_path)


print("\n======================================")
print("ALL 3 VISUALIZATIONS CREATED")
print("======================================")
print(f"Folder: {OUTPUT_DIR}")