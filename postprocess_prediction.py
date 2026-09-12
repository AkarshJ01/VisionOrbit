import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import rasterio
from rasterio.transform import xy
from pyproj import Transformer

from model import UNet


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/best_model_best_interval.pth"
PATCH_ROOT = "best_interval_patches"

THRESHOLD = 0.3

# Ignore extremely tiny isolated detections
MIN_REGION_PIXELS = 5

PATCH_SIZE = 256
PATCHES_PER_ROW = 4


# ============================================================
# FIND VALIDATION AOIs
# ============================================================

aois = sorted([
    d for d in os.listdir(PATCH_ROOT)
    if os.path.isdir(os.path.join(PATCH_ROOT, d))
])

validation_aois = aois[8:]

print("Using:", DEVICE)
print("Validation AOIs:", validation_aois)


# ============================================================
# FIND VALIDATION PATCHES
# ============================================================

samples = []

for aoi in validation_aois:

    aoi_dir = os.path.join(PATCH_ROOT, aoi)

    # Search recursively because patches are inside
    # interval directories.
    for root, dirs, files in os.walk(aoi_dir):

        for filename in sorted(files):

            if not filename.endswith("_old.npy"):
                continue

            index = filename.replace("_old.npy", "")

            old_path = os.path.join(
                root,
                f"{index}_old.npy"
            )

            new_path = os.path.join(
                root,
                f"{index}_new.npy"
            )

            mask_path = os.path.join(
                root,
                f"{index}_mask.npy"
            )

            if (
                os.path.exists(old_path)
                and os.path.exists(new_path)
                and os.path.exists(mask_path)
            ):
                samples.append(
                    (
                        aoi,
                        index,
                        old_path,
                        new_path,
                        mask_path
                    )
                )


# Sort deterministically by AOI and patch index
samples.sort(
    key=lambda s: (
        s[0],
        int(s[1])
    )
)

print("Validation patches:", len(samples))


# ============================================================
# LOAD CNN
# ============================================================

model = UNet(
    in_channels=8
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

print()
print("Model loaded.")


# ============================================================
# PROCESS ONE PATCH
# ============================================================

def process_patch(
    model,
    aoi,
    index,
    old_path,
    new_path,
    mask_path
):

    # --------------------------------------------------------
    # LOAD PATCH DATA
    # --------------------------------------------------------

    old = np.load(
        old_path
    ).astype(np.float32) / 255.0

    new = np.load(
        new_path
    ).astype(np.float32) / 255.0

    ground_truth = np.load(
        mask_path
    )

    # 4 old bands + 4 new bands
    image = np.concatenate(
        [old, new],
        axis=0
    )

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # LOAD GEOTIFF
    # --------------------------------------------------------

    tif_path = os.path.join(
        "data",
        aoi,
        "2019_12.tif"
    )

    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"Could not find GeoTIFF: {tif_path}"
        )

    with rasterio.open(tif_path) as src:

        transform = src.transform
        source_crs = src.crs
        image_width = src.width
        image_height = src.height

        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)

        pixel_area_m2 = (
            pixel_width * pixel_height
        )

        crs = str(src.crs)

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True
    )

    # --------------------------------------------------------
    # AUTOMATIC PATCH OFFSET
    # --------------------------------------------------------
    #
    # Patches are numbered row-major:
    #
    #  0   1   2   3
    #  4   5   6   7
    #  8   9  10  11
    # 12  13  14  15
    #
    # Therefore:
    #
    # column = index % 4
    # row    = index // 4
    #
    # and each patch is 256 x 256 pixels.
    # --------------------------------------------------------

    patch_index = int(index)

    patch_column = patch_index % PATCHES_PER_ROW
    patch_row = patch_index // PATCHES_PER_ROW

    patch_x_offset = (
        patch_column * PATCH_SIZE
    )

    patch_y_offset = (
        patch_row * PATCH_SIZE
    )

    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    prediction = (
        probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    ground_truth_pixels = int(
        ground_truth.sum()
    )

    predicted_pixels = int(
        prediction.sum()
    )

    total_predicted_area = (
        predicted_pixels *
        pixel_area_m2
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    # --------------------------------------------------------

    labeled, num_regions = ndimage.label(
        prediction
    )

    regions = []

    for region_id in range(
        1,
        num_regions + 1
    ):

        region_mask = (
            labeled == region_id
        )

        pixel_count = int(
            region_mask.sum()
        )

        # Ignore tiny noise
        if pixel_count < MIN_REGION_PIXELS:
            continue

        # Pixel coordinates inside THIS patch
        ys, xs = np.where(
            region_mask
        )

        # ----------------------------------------------------
        # PATCH-LOCAL BOUNDING BOX
        # ----------------------------------------------------

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        # ----------------------------------------------------
        # PATCH-LOCAL CENTROID
        # ----------------------------------------------------

        centroid_x = float(
            xs.mean()
        )

        centroid_y = float(
            ys.mean()
        )

        # ----------------------------------------------------
        # FULL-IMAGE COORDINATES
        # ----------------------------------------------------

        full_x = (
            centroid_x +
            patch_x_offset
        )

        full_y = (
            centroid_y +
            patch_y_offset
        )

        # Full-image bounding box
        full_x_min = (
            x_min +
            patch_x_offset
        )

        full_x_max = (
            x_max +
            patch_x_offset
        )

        full_y_min = (
            y_min +
            patch_y_offset
        )

        full_y_max = (
            y_max +
            patch_y_offset
        )

        # ----------------------------------------------------
        # PIXEL -> MAP COORDINATES
        # ----------------------------------------------------

        map_x, map_y = xy(
            transform,
            full_y,
            full_x,
            offset="center"
        )

        # ----------------------------------------------------
        # MAP -> LAT/LON
        # ----------------------------------------------------

        longitude, latitude = (
            transformer.transform(
                map_x,
                map_y
            )
        )

        # ----------------------------------------------------
        # PHYSICAL AREA
        # ----------------------------------------------------

        area_m2 = (
            pixel_count *
            pixel_area_m2
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        mean_confidence = float(
            probability[region_mask].mean()
        )

        max_confidence = float(
            probability[region_mask].max()
        )

        region = {

            "id": len(regions) + 1,

            "pixels": pixel_count,

            "area_m2": round(
                area_m2,
                2
            ),

            "bounding_box_patch_pixels": {

                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max

            },

            "bounding_box_full_image_pixels": {

                "x_min": full_x_min,
                "y_min": full_y_min,
                "x_max": full_x_max,
                "y_max": full_y_max

            },

            "centroid_patch_pixels": {

                "x": round(
                    centroid_x,
                    2
                ),

                "y": round(
                    centroid_y,
                    2
                )

            },

            "centroid_full_image_pixels": {

                "x": round(
                    full_x,
                    2
                ),

                "y": round(
                    full_y,
                    2
                )

            },

            "centroid_latitude": round(
                latitude,
                6
            ),

            "centroid_longitude": round(
                longitude,
                6
            ),

            "mean_confidence": round(
                mean_confidence,
                4
            ),

            "max_confidence": round(
                max_confidence,
                4
            )
        }

        regions.append(
            region
        )

    # --------------------------------------------------------
    # RETURN PATCH RESULTS
    # --------------------------------------------------------

    return {

        "aoi": aoi,

        "patch_index": patch_index,

        "patch_row": patch_row,

        "patch_column": patch_column,

        "patch_offset_pixels": {

            "x": patch_x_offset,
            "y": patch_y_offset

        },

        "patch_size_pixels": PATCH_SIZE,

        "time_period": {

            "before": "2018-01",
            "after": "2019-12"

        },

        "change_type":
            "likely new building / urban development",

        "model":
            "SpaceNet 7 U-Net",

        "threshold": THRESHOLD,

        "source_crs": crs,

        "output_crs": "EPSG:4326",

        "source_image_size_pixels": {

            "width": image_width,
            "height": image_height

        },

        "pixel_width_m": round(
            pixel_width,
            6
        ),

        "pixel_height_m": round(
            pixel_height,
            6
        ),

        "pixel_area_m2": round(
            pixel_area_m2,
            4
        ),

        "ground_truth_changed_pixels":
            ground_truth_pixels,

        "total_predicted_pixels":
            predicted_pixels,

        "total_predicted_area_m2":
            round(
                total_predicted_area,
                2
            ),

        "maximum_probability":
            float(probability.max()),

        "mean_probability":
            float(probability.mean()),

        "detected_regions":
            regions
    }


# ============================================================
# PROCESS ALL VALIDATION PATCHES
# ============================================================

all_results = []

print()
print("=" * 60)
print("PROCESSING ALL VALIDATION PATCHES")
print("=" * 60)

for counter, sample in enumerate(
    samples,
    start=1
):

    aoi, index, old_path, new_path, mask_path = sample

    print(
        f"[{counter}/{len(samples)}] "
        f"{aoi} | patch {index}"
    )

    result = process_patch(
        model,
        aoi,
        index,
        old_path,
        new_path,
        mask_path
    )

    all_results.append(
        result
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

total_ground_truth_pixels = sum(
    r["ground_truth_changed_pixels"]
    for r in all_results
)

total_predicted_pixels = sum(
    r["total_predicted_pixels"]
    for r in all_results
)

total_predicted_area = sum(
    r["total_predicted_area_m2"]
    for r in all_results
)

total_regions = sum(
    len(r["detected_regions"])
    for r in all_results
)

# Average confidence over all predicted pixels.
# Weight each patch by its number of predicted pixels.
confidence_numerator = 0.0

for r in all_results:

    confidence_numerator += (
        r["mean_probability"] *
        r["total_predicted_pixels"]
    )

if total_predicted_pixels > 0:

    overall_mean_probability = (
        confidence_numerator /
        total_predicted_pixels
    )

else:

    overall_mean_probability = 0.0


# ============================================================
# FINAL STRUCTURED OUTPUT
# ============================================================

results = {

    "project":
        "SpaceNet 7 long-term urban development change detection",

    "model":
        "SpaceNet 7 U-Net",

    "change_type":
        "likely new building / urban development",

    "time_period": {

        "before": "2018-01",
        "after": "2019-12"

    },

    "threshold": THRESHOLD,

    "patch_size_pixels": PATCH_SIZE,

    "source_crs":
        all_results[0]["source_crs"]
        if all_results else None,

    "output_crs": "EPSG:4326",

    "validation_aois":
        validation_aois,

    "validation_patch_count":
        len(all_results),

    "summary": {

        "ground_truth_changed_pixels":
            total_ground_truth_pixels,

        "predicted_changed_pixels":
            total_predicted_pixels,

        "predicted_changed_area_m2":
            round(
                total_predicted_area,
                2
            ),

        "detected_change_regions":
            total_regions,

        "mean_prediction_probability":
            round(
                overall_mean_probability,
                6
            )

    },

    "patch_results":
        all_results
}


json_file = (
    "all_validation_change_results.json"
)

with open(
    json_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(
    "Validation patches:",
    len(all_results)
)

print(
    "Ground truth changed pixels:",
    total_ground_truth_pixels
)

print(
    "Predicted changed pixels:",
    total_predicted_pixels
)

print(
    "Predicted changed area:",
    round(
        total_predicted_area,
        2
    ),
    "m²"
)

print(
    "Detected change regions:",
    total_regions
)

print(
    "Mean prediction probability:",
    round(
        overall_mean_probability,
        6
    )
)

print("=" * 60)

print()
print(
    "Saved structured results to:",
    json_file
)


# ============================================================
# VISUALIZE THE FIRST PATCH WITH CHANGE
# ============================================================

selected = None

for sample in samples:

    mask = np.load(
        sample[4]
    )

    if mask.sum() > 0:

        selected = sample
        break


if selected is not None:

    aoi, index, old_path, new_path, mask_path = selected

    old = (
        np.load(old_path)
        .astype(np.float32) / 255.0
    )

    new = (
        np.load(new_path)
        .astype(np.float32) / 255.0
    )

    selected_result = next(
        r for r in all_results
        if (
            r["aoi"] == aoi
            and r["patch_index"] == int(index)
        )
    )

    old_rgb = np.transpose(
        old[:3],
        (1, 2, 0)
    )

    new_rgb = np.transpose(
        new[:3],
        (1, 2, 0)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )

    # BEFORE
    axes[0].imshow(
        old_rgb
    )

    axes[0].set_title(
        "Before (2018)"
    )

    axes[0].axis("off")

    # AFTER
    axes[1].imshow(
        new_rgb
    )

    axes[1].set_title(
        "After (2019) + Detected Changes"
    )

    axes[1].axis("off")

    # Draw detected regions
    for region in selected_result[
        "detected_regions"
    ]:

        bbox = region[
            "bounding_box_patch_pixels"
        ]

        x_min = bbox["x_min"]
        y_min = bbox["y_min"]

        x_max = bbox["x_max"]
        y_max = bbox["y_max"]

        width = (
            x_max -
            x_min +
            1
        )

        height = (
            y_max -
            y_min +
            1
        )

        rectangle = plt.Rectangle(

            (x_min, y_min),

            width,
            height,

            fill=False,

            linewidth=2

        )

        axes[1].add_patch(
            rectangle
        )

        axes[1].text(

            x_min,

            max(
                0,
                y_min - 3
            ),

            f"Change {region['id']}",

            fontsize=9,

            backgroundcolor="white"

        )

    plt.tight_layout()

    output_file = (
        "change_regions.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved visualization to:",
        output_file
    )
