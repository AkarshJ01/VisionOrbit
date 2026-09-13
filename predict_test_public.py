import os
import json
import glob
import numpy as np
import torch
import rasterio
from scipy import ndimage
from pyproj import Transformer

from model import UNet


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = "checkpoints/best_model_best_interval.pth"

THRESHOLD = 0.3

PATCH_SIZE = 256
PATCHES_PER_ROW = 4

# Same filtering values used in your final validation pipeline
MERGE_DISTANCE_PIXELS = 20
MIN_REGION_PIXELS = 20
MIN_MEAN_CONFIDENCE = 0.60

TEST_ROOT = "data/test_public"

OUTPUT_JSON = "test_public_results.json"


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "aoi": "L15-0369E-1244N_1479_3214_13",
        "old": "2018_02",
        "new": "2019_12",
    },
    {
        "aoi": "L15-0509E-1108N_2037_3758_13",
        "old": "2018_02",
        "new": "2019_12",
    },
    {
        "aoi": "L15-1203E-1203N_4815_3379_13",
        "old": "2018_01",
        "new": "2019_09",
    },
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SPACE NET 7 — UNSEEN TEST")
print("=" * 70)

print(f"Using device: {DEVICE}")

model = UNet(in_channels=8).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded.")
print()


# ============================================================
# FIND TIFF
# ============================================================

def find_tif(aoi, month):

    pattern = os.path.join(
        TEST_ROOT,
        aoi,
        "images_masked",
        f"global_monthly_{month}_mosaic_{aoi}.tif"
    )

    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"Could not find:\n{pattern}"
        )

    return matches[0]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    with rasterio.open(path) as src:

        image = src.read()

        transform = src.transform
        source_crs = src.crs
        width = src.width
        height = src.height

        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)

        pixel_area = (
            pixel_width *
            pixel_height
        )

    if image.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands, got {image.shape[0]}"
        )

    return (
        image,
        transform,
        source_crs,
        width,
        height,
        pixel_area
    )


# ============================================================
# MERGE REGIONS
# ============================================================

def bbox_distance(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = max(
        0,
        max(ax1, bx1) -
        min(ax2, bx2)
    )

    dy = max(
        0,
        max(ay1, by1) -
        min(ay2, by2)
    )

    return max(dx, dy)


def merge_regions(regions):

    groups = []

    for region in regions:

        added = False

        for group in groups:

            group_bbox = group["bbox"]

            if bbox_distance(
                region["bbox"],
                group_bbox
            ) <= MERGE_DISTANCE_PIXELS:

                group["regions"].append(region)

                # Update group bounding box
                xs = []
                ys = []

                for r in group["regions"]:

                    x1, y1, x2, y2 = r["bbox"]

                    xs.extend([x1, x2])
                    ys.extend([y1, y2])

                group["bbox"] = [
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys)
                ]

                added = True
                break

        if not added:

            groups.append({
                "regions": [region],
                "bbox": region["bbox"]
            })

    # --------------------------------------------------------
    # Repeat until no more groups can merge
    # --------------------------------------------------------

    changed = True

    while changed:

        changed = False
        new_groups = []

        while groups:

            current = groups.pop(0)

            merged = False

            for i, other in enumerate(groups):

                if bbox_distance(
                    current["bbox"],
                    other["bbox"]
                ) <= MERGE_DISTANCE_PIXELS:

                    combined_regions = (
                        current["regions"] +
                        other["regions"]
                    )

                    xs = []
                    ys = []

                    for r in combined_regions:

                        x1, y1, x2, y2 = r["bbox"]

                        xs.extend([x1, x2])
                        ys.extend([y1, y2])

                    combined_bbox = [
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys)
                    ]

                    groups[i] = {
                        "regions": combined_regions,
                        "bbox": combined_bbox
                    }

                    merged = True
                    changed = True
                    break

            if not merged:
                new_groups.append(current)

        groups = new_groups

    return groups


# ============================================================
# PROCESS ONE TEST CASE
# ============================================================

def process_test_case(case):

    aoi = case["aoi"]
    old_month = case["old"]
    new_month = case["new"]

    print("=" * 70)
    print(f"AOI: {aoi}")
    print(f"Period: {old_month} -> {new_month}")
    print("=" * 70)

    old_path = find_tif(
        aoi,
        old_month
    )

    new_path = find_tif(
        aoi,
        new_month
    )

    print("Old image:")
    print(old_path)

    print("New image:")
    print(new_path)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    old_image, transform, source_crs, width, height, pixel_area = (
        load_image(old_path)
    )

    new_image, _, _, new_width, new_height, _ = (
        load_image(new_path)
    )

    if (
        width != new_width or
        height != new_height
    ):
        raise ValueError(
            "Old and new images have different dimensions."
        )

    if old_image.shape != new_image.shape:
        raise ValueError(
            "Old and new images have different shapes."
        )

    print()
    print(f"Image size: {width} x {height}")
    print(f"Bands: {old_image.shape[0]}")
    print(f"CRS: {source_crs}")
    print(f"Pixel area: {pixel_area:.4f} m²")

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    old = (
        old_image.astype(np.float32) /
        255.0
    )

    new = (
        new_image.astype(np.float32) /
        255.0
    )

    # --------------------------------------------------------
    # Full-resolution probability map
    # --------------------------------------------------------

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Run 256x256 patches
    # --------------------------------------------------------

    for patch_row in range(
        height // PATCH_SIZE
    ):

        for patch_col in range(
            width // PATCH_SIZE
        ):

            y1 = (
                patch_row *
                PATCH_SIZE
            )

            y2 = y1 + PATCH_SIZE

            x1 = (
                patch_col *
                PATCH_SIZE
            )

            x2 = x1 + PATCH_SIZE

            old_patch = old[
                :,
                y1:y2,
                x1:x2
            ]

            new_patch = new[
                :,
                y1:y2,
                x1:x2
            ]

            # 4 old + 4 new = 8 channels
            combined = np.concatenate(
                [
                    old_patch,
                    new_patch
                ],
                axis=0
            )

            tensor = torch.from_numpy(
                combined
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                logits = model(
                    tensor
                )

                probability = torch.sigmoid(
                    logits
                )[0, 0].cpu().numpy()

            probability_map[
                y1:y2,
                x1:x2
            ] = probability

            patch_index = (
                patch_row *
                PATCHES_PER_ROW +
                patch_col
            )

            print(
                f"  Processed patch {patch_index:02d}"
            )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    prediction = (
        probability_map >= THRESHOLD
    )

    predicted_pixels = int(
        prediction.sum()
    )

    predicted_area = (
        predicted_pixels *
        pixel_area
    )

    print()
    print(
        f"Predicted changed pixels: "
        f"{predicted_pixels}"
    )

    print(
        f"Predicted changed area: "
        f"{predicted_area:.2f} m²"
    )

    print(
        f"Maximum probability: "
        f"{probability_map.max():.4f}"
    )

    print(
        f"Mean probability: "
        f"{probability_map.mean():.6f}"
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    labeled, num_regions = ndimage.label(
        prediction
    )

    raw_regions = []

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

        if pixel_count < MIN_REGION_PIXELS:
            continue

        ys, xs = np.where(
            region_mask
        )

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1

        y1 = int(ys.min())
        y2 = int(ys.max()) + 1

        confidence_values = (
            probability_map[
                region_mask
            ]
        )

        mean_confidence = float(
            confidence_values.mean()
        )

        max_confidence = float(
            confidence_values.max()
        )

        raw_regions.append({
            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],
            "pixels": pixel_count,
            "area_m2": (
                pixel_count *
                pixel_area
            ),
            "mean_confidence": mean_confidence,
            "max_confidence": max_confidence
        })

    print(
        f"Raw regions after size filter: "
        f"{len(raw_regions)}"
    )

    # --------------------------------------------------------
    # Merge nearby regions
    # --------------------------------------------------------

    groups = merge_regions(
        raw_regions
    )

    print(
        f"Merged regions: "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Convert merged groups into events
    # --------------------------------------------------------

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True
    )

    final_events = []

    for event_number, group in enumerate(
        groups,
        start=1
    ):

        regions = group["regions"]

        total_pixels = sum(
            r["pixels"]
            for r in regions
        )

        total_area = (
            total_pixels *
            pixel_area
        )

        # Pixel-weighted confidence
        mean_confidence = (
            sum(
                r["mean_confidence"] *
                r["pixels"]
                for r in regions
            )
            / total_pixels
        )

        max_confidence = max(
            r["max_confidence"]
            for r in regions
        )

        # Filter low-confidence merged regions
        if (
            total_pixels <
            MIN_REGION_PIXELS
        ):
            continue

        if (
            mean_confidence <
            MIN_MEAN_CONFIDENCE
        ):
            continue

        x1, y1, x2, y2 = group["bbox"]

        # Center pixel
        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        lon, lat = rasterio.transform.xy(
            transform,
            center_y,
            center_x,
            offset="center"
        )

        # xy returns coordinates in source CRS
        lon, lat = transformer.transform(
            lon,
            lat
        )

        final_events.append({
            "event_id": (
                f"{aoi}_{event_number:03d}"
            ),
            "aoi": aoi,
            "time_period": (
                f"{old_month} to {new_month}"
            ),
            "change_type": (
                "likely new building / "
                "urban development"
            ),
            "latitude": float(lat),
            "longitude": float(lon),
            "area_m2": round(
                total_area,
                2
            ),
            "detected_pixels": total_pixels,
            "mean_confidence": round(
                mean_confidence,
                4
            ),
            "max_confidence": round(
                max_confidence,
                4
            ),
            "bbox_pixels": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        })

    print(
        f"Final change events: "
        f"{len(final_events)}"
    )

    return {
        "aoi": aoi,
        "old_month": old_month,
        "new_month": new_month,
        "image_width": width,
        "image_height": height,
        "source_crs": str(source_crs),
        "threshold": THRESHOLD,
        "predicted_changed_pixels": predicted_pixels,
        "predicted_changed_area_m2": round(
            predicted_area,
            2
        ),
        "raw_regions": len(raw_regions),
        "merged_regions": len(groups),
        "final_change_events": len(
            final_events
        ),
        "events": final_events
    }


# ============================================================
# RUN ALL TESTS
# ============================================================

all_results = []

for case in TEST_CASES:

    result = process_test_case(
        case
    )

    all_results.append(
        result
    )

    print()


# ============================================================
# SAVE JSON
# ============================================================

output = {
    "project": (
        "SpaceNet 7 long-term urban "
        "development change detection"
    ),
    "model": "SpaceNet 7 U-Net",
    "test_type": (
        "unseen test_public imagery"
    ),
    "model_checkpoint": CHECKPOINT,
    "threshold": THRESHOLD,
    "merge_distance_pixels": (
        MERGE_DISTANCE_PIXELS
    ),
    "minimum_region_pixels": (
        MIN_REGION_PIXELS
    ),
    "minimum_mean_confidence": (
        MIN_MEAN_CONFIDENCE
    ),
    "tests": all_results
}

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(
    f"Saved results to: {OUTPUT_JSON}"
)
