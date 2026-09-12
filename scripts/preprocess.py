from pathlib import Path
import json
import re

import numpy as np
import rasterio
from rasterio.features import rasterize


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("raw")

DATASET_DIR = next(
    ROOT.glob("SN6_buildings_AOI_11_Rotterdam_train_sample/AOI_11_Rotterdam")
)

RGB_DIR = DATASET_DIR / "PS-RGB"
SAR_DIR = DATASET_DIR / "SAR-Intensity"
GEOJSON_DIR = DATASET_DIR / "geojson_buildings"

OUTPUT_DIR = Path("processed")
IMAGE_DIR = OUTPUT_DIR / "images"
MASK_DIR = OUTPUT_DIR / "masks"

PATCH_SIZE = 256
STRIDE = 224


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER: GET TILE NUMBER
# ============================================================

def get_tile_number(filename):
    """
    Extract tile number from a SpaceNet filename.

    Example:
        ..._tile_69.tif
        -> 69
    """

    match = re.search(r"_tile_(\d+)", filename)

    if match is None:
        return None

    return match.group(1)


# ============================================================
# HELPER: NORMALIZE SAR
# ============================================================

def normalize_sar(sar):
    """
    Normalize each SAR channel independently.

    Uses robust 2nd and 98th percentiles so that
    extreme values do not dominate the image.
    """

    sar = sar.astype(np.float32)

    normalized = np.zeros_like(sar, dtype=np.float32)

    for band in range(sar.shape[0]):

        channel = sar[band]

        low = np.percentile(channel, 2)
        high = np.percentile(channel, 98)

        if high <= low:
            normalized[band] = 0
        else:
            normalized[band] = np.clip(
                (channel - low) / (high - low),
                0,
                1
            )

    return normalized


# ============================================================
# HELPER: CREATE BUILDING MASK
# ============================================================

def create_building_mask(geojson_path, raster_shape, transform):

    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    geometries = []

    for feature in geojson["features"]:

        geometry = feature.get("geometry")

        if geometry is not None:
            geometries.append(geometry)

    mask = rasterize(
        geometries,
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )

    return mask


# ============================================================
# FIND ALL RGB FILES
# ============================================================

rgb_files = sorted(RGB_DIR.glob("*.tif"))

print("=" * 60)
print("MULTI-SENSOR PREPROCESSING")
print("=" * 60)

print(f"RGB files found: {len(rgb_files)}")
print()


# ============================================================
# PROCESS EACH TILE
# ============================================================

total_patches = 0
processed_tiles = 0


for rgb_path in rgb_files:

    tile_number = get_tile_number(rgb_path.name)

    if tile_number is None:
        print(f"Skipping unknown tile: {rgb_path.name}")
        continue


    # --------------------------------------------------------
    # FIND MATCHING SAR
    # --------------------------------------------------------

    sar_candidates = list(
        SAR_DIR.glob(f"*tile_{tile_number}.tif")
    )

    if len(sar_candidates) == 0:

        print(
            f"[WARNING] No SAR file found for tile {tile_number}"
        )

        continue

    sar_path = sar_candidates[0]


    # --------------------------------------------------------
    # FIND MATCHING GEOJSON
    # --------------------------------------------------------

    geojson_candidates = list(
        GEOJSON_DIR.glob(f"*tile_{tile_number}.geojson")
    )

    if len(geojson_candidates) == 0:

        print(
            f"[WARNING] No GeoJSON found for tile {tile_number}"
        )

        continue

    geojson_path = geojson_candidates[0]


    print("-" * 60)
    print(f"Processing tile {tile_number}")
    print(f"RGB:     {rgb_path.name}")
    print(f"SAR:     {sar_path.name}")
    print(f"Labels:  {geojson_path.name}")


    # --------------------------------------------------------
    # READ RGB
    # --------------------------------------------------------

    with rasterio.open(rgb_path) as src:

        rgb = src.read()

        rgb_transform = src.transform
        rgb_crs = src.crs

        width = src.width
        height = src.height


    # --------------------------------------------------------
    # READ SAR
    # --------------------------------------------------------

    with rasterio.open(sar_path) as src:

        sar = src.read()

        sar_transform = src.transform
        sar_crs = src.crs


    # --------------------------------------------------------
    # CHECK DIMENSIONS
    # --------------------------------------------------------

    if rgb.shape[1:] != sar.shape[1:]:

        print(
            f"[WARNING] RGB/SAR dimensions do not match "
            f"for tile {tile_number}"
        )

        continue


    # --------------------------------------------------------
    # CHECK CRS
    # --------------------------------------------------------

    if rgb_crs != sar_crs:

        print(
            f"[WARNING] RGB/SAR CRS mismatch "
            f"for tile {tile_number}"
        )

        continue


    # --------------------------------------------------------
    # NORMALIZE RGB
    # --------------------------------------------------------

    rgb = rgb.astype(np.float32) / 255.0


    # --------------------------------------------------------
    # NORMALIZE SAR
    # --------------------------------------------------------

    sar = normalize_sar(sar)


    # --------------------------------------------------------
    # CREATE BUILDING MASK
    # --------------------------------------------------------

    mask = create_building_mask(
        geojson_path,
        (height, width),
        rgb_transform
    )


    # --------------------------------------------------------
    # STACK RGB + SAR
    # --------------------------------------------------------

    # RGB:
    #   3 channels
    #
    # SAR:
    #   4 channels
    #
    # Combined:
    #   7 channels

    combined = np.concatenate(
        [rgb, sar],
        axis=0
    )


    print(f"Combined shape: {combined.shape}")
    print(f"Mask shape:     {mask.shape}")
    print(f"Building pixels: {mask.sum()}")


    # ========================================================
    # CREATE PATCHES
    # ========================================================

    tile_patch_count = 0

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            image_patch = combined[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            mask_patch = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]


            # ------------------------------------------------
            # SAVE PATCH
            # ------------------------------------------------

            patch_name = (
                f"tile_{tile_number}"
                f"_y{y}"
                f"_x{x}"
            )

            np.save(
                IMAGE_DIR / f"{patch_name}.npy",
                image_patch.astype(np.float32)
            )

            np.save(
                MASK_DIR / f"{patch_name}.npy",
                mask_patch.astype(np.uint8)
            )


            tile_patch_count += 1
            total_patches += 1


    print(f"Patches created: {tile_patch_count}")

    processed_tiles += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Tiles processed: {processed_tiles}")
print(f"Total patches:   {total_patches}")

print()
print(f"Images saved to: {IMAGE_DIR}")
print(f"Masks saved to:  {MASK_DIR}")
print()
print("Each image has:")
print("  3 RGB channels")
print("  4 SAR channels")
print("  ----------------")
print("  7 total channels")