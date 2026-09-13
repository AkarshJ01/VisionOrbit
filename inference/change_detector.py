import os
import sys
from pathlib import Path
import numpy as np
import torch
import rasterio
from scipy import ndimage
from pyproj import Transformer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

DEFAULT_CHECKPOINT = os.path.join(str(PROJECT_ROOT), "checkpoints", "best_model_best_interval.pth")
CHECKPOINT = DEFAULT_CHECKPOINT if os.path.exists(DEFAULT_CHECKPOINT) else "checkpoints/best_model_best_interval.pth"

THRESHOLD = 0.3

PATCH_SIZE = 256

MERGE_DISTANCE_PIXELS = 20
MIN_REGION_PIXELS = 20
MIN_MEAN_CONFIDENCE = 0.60


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = UNet(
        in_channels=8
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


# Load once when this module is imported
model = load_model()


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

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
            f"{path}: expected 4 bands, "
            f"got {image.shape[0]}"
        )

    return {
        "image": image,
        "transform": transform,
        "crs": source_crs,
        "width": width,
        "height": height,
        "pixel_area": pixel_area
    }


# ============================================================
# VALIDATE INPUT IMAGES
# ============================================================

def validate_images(old_data, new_data):

    if (
        old_data["width"] !=
        new_data["width"]
        or
        old_data["height"] !=
        new_data["height"]
    ):

        raise ValueError(
            "Old and new images must have "
            "the same dimensions."
        )

    if old_data["crs"] != new_data["crs"]:

        raise ValueError(
            "Old and new images must use "
            "the same CRS."
        )

    old_transform = old_data["transform"]
    new_transform = new_data["transform"]

    if not np.allclose(
        np.array(old_transform),
        np.array(new_transform)
    ):

        raise ValueError(
            "Old and new images must have "
            "the same spatial alignment."
        )

    if not np.isclose(
        old_data["pixel_area"],
        new_data["pixel_area"]
    ):

        raise ValueError(
            "Old and new images must have "
            "the same spatial resolution."
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

            if bbox_distance(
                region["bbox"],
                group["bbox"]
            ) <= MERGE_DISTANCE_PIXELS:

                group["regions"].append(region)

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

    # Repeat merging until stable
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

                    groups[i] = {
                        "regions": combined_regions,
                        "bbox": [
                            min(xs),
                            min(ys),
                            max(xs),
                            max(ys)
                        ]
                    }

                    merged = True
                    changed = True

                    break

            if not merged:

                new_groups.append(current)

        groups = new_groups

    return groups


# ============================================================
# CREATE PATCH RANGES
# ============================================================

def generate_patch_ranges(width, height):

    patches = []

    for y1 in range(
        0,
        height,
        PATCH_SIZE
    ):

        for x1 in range(
            0,
            width,
            PATCH_SIZE
        ):

            x2 = min(
                x1 + PATCH_SIZE,
                width
            )

            y2 = min(
                y1 + PATCH_SIZE,
                height
            )

            patch_width = x2 - x1
            patch_height = y2 - y1

            # The CNN was trained on exactly
            # 256 x 256 patches.
            #
            # Edge patches smaller than 256 are
            # padded and then cropped back.

            patches.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": patch_width,
                "height": patch_height
            })

    return patches


# ============================================================
# RUN CNN
# ============================================================

def predict_probability_map(
    old_image,
    new_image
):

    height = old_image.shape[1]
    width = old_image.shape[2]

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    patches = generate_patch_ranges(
        width,
        height
    )

    for patch in patches:

        x1 = patch["x1"]
        y1 = patch["y1"]
        x2 = patch["x2"]
        y2 = patch["y2"]

        patch_width = patch["width"]
        patch_height = patch["height"]

        old_patch = old_image[
            :,
            y1:y2,
            x1:x2
        ]

        new_patch = new_image[
            :,
            y1:y2,
            x1:x2
        ]

        # ----------------------------------------------------
        # Pad edge patches to 256 x 256
        # ----------------------------------------------------

        old_padded = np.zeros(
            (
                4,
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.float32
        )

        new_padded = np.zeros(
            (
                4,
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.float32
        )

        old_padded[
            :,
            :patch_height,
            :patch_width
        ] = old_patch

        new_padded[
            :,
            :patch_height,
            :patch_width
        ] = new_patch

        # ----------------------------------------------------
        # 4 old bands + 4 new bands = 8 channels
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                old_padded,
                new_padded
            ],
            axis=0
        )

        tensor = torch.from_numpy(
            combined
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # CNN inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

        # ----------------------------------------------------
        # Remove padding
        # ----------------------------------------------------

        probability_map[
            y1:y2,
            x1:x2
        ] = probability[
            :patch_height,
            :patch_width
        ]

    return probability_map


# ============================================================
# EXTRACT REGIONS
# ============================================================

def extract_regions(
    probability_map,
    pixel_area
):

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

    # Connected components
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
            "mean_confidence": (
                mean_confidence
            ),
            "max_confidence": (
                max_confidence
            )
        })

    return (
        prediction,
        predicted_pixels,
        predicted_area,
        raw_regions
    )


# ============================================================
# CONVERT REGIONS TO EVENTS
# ============================================================

def create_events(
    groups,
    transform,
    source_crs,
    pixel_area
):

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True
    )

    events = []

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
                r["mean_confidence"]
                * r["pixels"]
                for r in regions
            )
            / total_pixels
        )

        max_confidence = max(
            r["max_confidence"]
            for r in regions
        )

        # Confidence filter
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

        x1, y1, x2, y2 = (
            group["bbox"]
        )

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        # Convert pixel coordinates
        # to source CRS coordinates
        x, y = rasterio.transform.xy(
            transform,
            center_y,
            center_x,
            offset="center"
        )

        # Convert source CRS → WGS84
        lon, lat = transformer.transform(
            x,
            y
        )

        events.append({
            "event_id": (
                f"change_{event_number:03d}"
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

            "detected_pixels": (
                total_pixels
            ),

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

    return events


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def detect_change(
    old_image_path,
    new_image_path
):

    print("=" * 70)
    print("SPACE NET 7 — CHANGE DETECTION")
    print("=" * 70)

    print(f"Old image: {old_image_path}")
    print(f"New image: {new_image_path}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    old_data = load_image(
        old_image_path
    )

    new_data = load_image(
        new_image_path
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_images(
        old_data,
        new_data
    )

    old_image = (
        old_data["image"]
    )

    new_image = (
        new_data["image"]
    )

    width = old_data["width"]
    height = old_data["height"]

    pixel_area = (
        old_data["pixel_area"]
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    old_image = (
        old_image.astype(
            np.float32
        ) / 255.0
    )

    new_image = (
        new_image.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    print("Running CNN...")

    probability_map = (
        predict_probability_map(
            old_image,
            new_image
        )
    )

    # --------------------------------------------------------
    # Extract regions
    # --------------------------------------------------------

    (
        prediction,
        predicted_pixels,
        predicted_area,
        raw_regions
    ) = extract_regions(
        probability_map,
        pixel_area
    )

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

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    groups = merge_regions(
        raw_regions
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    events = create_events(
        groups,
        old_data["transform"],
        old_data["crs"],
        pixel_area
    )

    print(
        f"Raw regions: "
        f"{len(raw_regions)}"
    )

    print(
        f"Merged regions: "
        f"{len(groups)}"
    )

    print(
        f"Final events: "
        f"{len(events)}"
    )

    # --------------------------------------------------------
    # Return RAG-ready result
    # --------------------------------------------------------

    return {
        "model": "SpaceNet 7 U-Net",

        "input": {
            "old_image": old_image_path,
            "new_image": new_image_path
        },

        "image": {
            "width": width,
            "height": height,
            "source_crs": str(
                old_data["crs"]
            ),
            "pixel_area_m2": pixel_area
        },

        "processing": {
            "threshold": THRESHOLD,
            "merge_distance_pixels": (
                MERGE_DISTANCE_PIXELS
            ),
            "minimum_region_pixels": (
                MIN_REGION_PIXELS
            ),
            "minimum_mean_confidence": (
                MIN_MEAN_CONFIDENCE
            )
        },

        "summary": {
            "predicted_changed_pixels": (
                predicted_pixels
            ),

            "predicted_changed_area_m2": round(
                predicted_area,
                2
            ),

            "raw_regions": len(
                raw_regions
            ),

            "merged_regions": len(
                groups
            ),

            "final_change_events": len(
                events
            )
        },

        "events": events
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VisionOrbit Satellite Urban Change Detection")
    parser.add_argument("--before", type=str, default=None, help="Path to earlier 4-band GeoTIFF")
    parser.add_argument("--after", type=str, default=None, help="Path to later 4-band GeoTIFF")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON results")
    args = parser.parse_args()

    if args.before and args.after:
        print(f"Running change detection between:\n  Before: {args.before}\n  After:  {args.after}")
        results = detect_change(args.before, args.after, threshold=args.threshold)
    else:
        # Default test pair
        default_before = os.path.join(
            str(PROJECT_ROOT),
            "data/test_public/L15-0369E-1244N_1479_3214_13/images_masked/global_monthly_2018_02_mosaic_L15-0369E-1244N_1479_3214_13.tif"
        )
        default_after = os.path.join(
            str(PROJECT_ROOT),
            "data/test_public/L15-0369E-1244N_1479_3214_13/images_masked/global_monthly_2019_12_mosaic_L15-0369E-1244N_1479_3214_13.tif"
        )
        if os.path.exists(default_before) and os.path.exists(default_after):
            print("No images specified. Running on default test pair...")
            results = detect_change(default_before, default_after, threshold=args.threshold)
        else:
            print("Error: Please provide --before and --after image paths.")
            return

    print("\nSummary:", results.get("summary"))
    print(f"Found {len(results.get('events', []))} change events.")

    if args.output:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()