import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SETTINGS
# ============================================================

DATA_ROOT = os.path.join(str(PROJECT_ROOT), "data")
OUTPUT_ROOT = os.path.join(str(PROJECT_ROOT), "data/patches/long_interval")

# Long temporal intervals
PAIRS = [
    ("2018_01", "2019_12"),
    ("2018_06", "2019_12"),
    ("2019_01", "2019_12"),
    ("2019_06", "2019_12"),
]


# ============================================================
# FIND FILE
# ============================================================

def find_file(aoi, month, extension):
    """
    Find a SpaceNet 7 file.

    Jan 2018 and Dec 2019 were downloaded using simplified
    filenames, while intermediate months use the original
    SpaceNet 7 naming convention.
    """

    folder = os.path.join(DATA_ROOT, aoi)

    if not os.path.exists(folder):
        return None

    # --------------------------------------------------------
    # Check simplified filename first
    # Example:
    # 2018_01.tif
    # 2018_01_Buildings.geojson
    # --------------------------------------------------------

    simple_name = f"{month}{extension}"

    simple_path = os.path.join(folder, simple_name)

    if os.path.exists(simple_path):
        return simple_path

    # --------------------------------------------------------
    # Check original SpaceNet 7 filename
    # Example:
    # global_monthly_2018_06_mosaic_L15-....tif
    # --------------------------------------------------------

    for filename in os.listdir(folder):

        if not filename.endswith(extension):
            continue

        if not filename.startswith(
            f"global_monthly_{month}_"
        ):
            continue

        if aoi not in filename:
            continue

        return os.path.join(folder, filename)

    return None


# ============================================================
# CREATE CHANGE MASK
# ============================================================

def get_new_building_mask(
    old_geojson,
    new_geojson,
    shape
):
    """
    New buildings are defined as buildings whose persistent
    SpaceNet 7 ID exists in the later image but not the
    earlier image.
    """

    old_gdf = gpd.read_file(old_geojson)
    new_gdf = gpd.read_file(new_geojson)

    # Empty annotation files
    if len(old_gdf) == 0 or len(new_gdf) == 0:
        return None

    # Need persistent building IDs
    if (
        "Id" not in old_gdf.columns
        or "Id" not in new_gdf.columns
    ):
        return None

    # IDs in old image
    old_ids = set(old_gdf["Id"])

    # Buildings appearing only in the later image
    new_buildings = new_gdf[
        ~new_gdf["Id"].isin(old_ids)
    ]

    # No new buildings
    if len(new_buildings) == 0:

        return np.zeros(
            shape,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # labels_match_pix geometry is already in image
    # pixel coordinates.
    # Therefore we use an identity transform.
    # --------------------------------------------------------

    shapes = [
        (geom, 1)
        for geom in new_buildings.geometry
        if geom is not None
        and not geom.is_empty
    ]

    if len(shapes) == 0:

        return np.zeros(
            shape,
            dtype=np.uint8
        )

    mask = rasterize(
        shapes,
        out_shape=shape,
        transform=Affine.identity(),
        fill=0,
        dtype=np.uint8
    )

    return mask


# ============================================================
# CREATE 256x256 PATCHES
# ============================================================

def create_patches(
    old_img,
    new_img,
    mask,
    output_folder
):

    patch_size = 256

    # Image shape:
    # (bands, height, width)

    _, height, width = old_img.shape

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    patch_count = 0
    changed_count = 0

    # --------------------------------------------------------
    # Only use complete 256x256 patches.
    # --------------------------------------------------------

    for y in range(
        0,
        height - patch_size + 1,
        patch_size
    ):

        for x in range(
            0,
            width - patch_size + 1,
            patch_size
        ):

            old_patch = old_img[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            new_patch = new_img[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            mask_patch = mask[
                y:y + patch_size,
                x:x + patch_size
            ]

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if old_patch.shape != (
                4,
                patch_size,
                patch_size
            ):
                continue

            if new_patch.shape != (
                4,
                patch_size,
                patch_size
            ):
                continue

            # ------------------------------------------------
            # Save patches
            # ------------------------------------------------

            np.save(
                os.path.join(
                    output_folder,
                    f"{patch_count}_old.npy"
                ),
                old_patch
            )

            np.save(
                os.path.join(
                    output_folder,
                    f"{patch_count}_new.npy"
                ),
                new_patch
            )

            np.save(
                os.path.join(
                    output_folder,
                    f"{patch_count}_mask.npy"
                ),
                mask_patch
            )

            if mask_patch.sum() > 0:
                changed_count += 1

            patch_count += 1

    return (
        patch_count,
        changed_count
    )


# ============================================================
# FIND AOIs
# ============================================================

aois = sorted(
    d
    for d in os.listdir(DATA_ROOT)
    if os.path.isdir(
        os.path.join(DATA_ROOT, d)
    )
)

print(
    f"AOIs found: {len(aois)}"
)

print()


# ============================================================
# COUNTERS
# ============================================================

total_pairs = 0
skipped_pairs = 0

total_patches = 0
total_change_patches = 0

total_changed_pixels = 0
total_pixels = 0


# ============================================================
# PROCESS EACH AOI
# ============================================================

for aoi in aois:

    print("=" * 70)
    print(aoi)

    # --------------------------------------------------------
    # Process each long interval
    # --------------------------------------------------------

    for old_month, new_month in PAIRS:

        actual_old_month = old_month

        # ----------------------------------------------------
        # AOI 3 does not have January 2018.
        #
        # Use February 2018 instead.
        # ----------------------------------------------------

        if old_month == "2018_01":

            old_img_path = find_file(
                aoi,
                "2018_01",
                ".tif"
            )

            if old_img_path is None:

                old_img_path = find_file(
                    aoi,
                    "2018_02",
                    ".tif"
                )

                if old_img_path is not None:
                    actual_old_month = "2018_02"

        else:

            old_img_path = find_file(
                aoi,
                old_month,
                ".tif"
            )

        # ----------------------------------------------------
        # Find new image
        # ----------------------------------------------------

        new_img_path = find_file(
            aoi,
            new_month,
            ".tif"
        )

        # ----------------------------------------------------
        # Find annotations
        # ----------------------------------------------------

        old_geojson = find_file(
            aoi,
            actual_old_month,
            "_Buildings.geojson"
        )

        new_geojson = find_file(
            aoi,
            new_month,
            "_Buildings.geojson"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if (
            old_img_path is None
            or new_img_path is None
            or old_geojson is None
            or new_geojson is None
        ):

            print(
                f"  SKIP "
                f"{actual_old_month} -> {new_month}: "
                f"missing file"
            )

            skipped_pairs += 1

            continue

        print(
            f"  Processing "
            f"{actual_old_month} -> {new_month}"
        )

        # ----------------------------------------------------
        # Read old image
        # ----------------------------------------------------

        with rasterio.open(
            old_img_path
        ) as src:

            old_img = src.read()

        # ----------------------------------------------------
        # Read new image
        # ----------------------------------------------------

        with rasterio.open(
            new_img_path
        ) as src:

            new_img = src.read()

        # ----------------------------------------------------
        # Check image dimensions
        # ----------------------------------------------------

        if old_img.shape != new_img.shape:

            print(
                f"    SKIP: image shapes differ "
                f"{old_img.shape} vs "
                f"{new_img.shape}"
            )

            skipped_pairs += 1

            continue

        # ----------------------------------------------------
        # Create change mask
        # ----------------------------------------------------

        mask = get_new_building_mask(
            old_geojson,
            new_geojson,
            old_img.shape[1:]
        )

        # ----------------------------------------------------
        # Invalid / empty annotations
        # ----------------------------------------------------

        if mask is None:

            print(
                "    SKIP: "
                "invalid/empty annotation file"
            )

            skipped_pairs += 1

            continue

        # ----------------------------------------------------
        # Pair name
        # ----------------------------------------------------

        pair_name = (
            f"{actual_old_month}"
            f"_to_"
            f"{new_month}"
        )

        output_folder = os.path.join(
            OUTPUT_ROOT,
            aoi,
            pair_name
        )

        # ----------------------------------------------------
        # Create patches
        # ----------------------------------------------------

        patches, changed_patches = create_patches(
            old_img,
            new_img,
            mask,
            output_folder
        )

        changed_pixels = int(
            mask.sum()
        )

        pixels = (
            mask.shape[0]
            * mask.shape[1]
        )

        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        total_pairs += 1

        total_patches += patches

        total_change_patches += (
            changed_patches
        )

        total_changed_pixels += (
            changed_pixels
        )

        total_pixels += pixels

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"    Patches: {patches}"
            f" | Change patches: "
            f"{changed_patches}"
            f" | Changed pixels: "
            f"{changed_pixels}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 70)
print(
    "LONG-INTERVAL PATCH CREATION COMPLETE"
)
print("=" * 70)

print(
    f"Valid temporal pairs: "
    f"{total_pairs}"
)

print(
    f"Skipped pairs: "
    f"{skipped_pairs}"
)

print(
    f"Total patches: "
    f"{total_patches}"
)

print(
    f"Patches with change: "
    f"{total_change_patches}"
)

print(
    f"Changed pixels: "
    f"{total_changed_pixels}"
)

print(
    f"Total pixels: "
    f"{total_pixels}"
)

if total_patches > 0:

    print(
        f"Change-patch ratio: "
        f"{100 * total_change_patches / total_patches:.2f}%"
    )

if total_pixels > 0:

    print(
        f"Pixel change ratio: "
        f"{100 * total_changed_pixels / total_pixels:.4f}%"
    )