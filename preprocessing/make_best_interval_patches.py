import os
import sys
from pathlib import Path
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = os.path.join(str(PROJECT_ROOT), "data")
OUTPUT_ROOT = os.path.join(str(PROJECT_ROOT), "data/patches/best_interval")
PATCH_SIZE = 256


def find_file(aoi, month, extension):

    folder = os.path.join(DATA_ROOT, aoi)

    # Simplified filenames
    simple = os.path.join(
        folder,
        f"{month}{extension}"
    )

    if os.path.exists(simple):
        return simple

    # Full SpaceNet filenames
    pattern = os.path.join(
        folder,
        f"global_monthly_{month}_*{extension}"
    )

    matches = glob.glob(pattern)

    if matches:
        return matches[0]

    return None


def get_start_month(aoi):

    # AOI 3 has no January 2018
    if aoi == "L15-0358E-1220N_1433_3310_13":
        return "2018_02"

    return "2018_01"


def create_patches(aoi, old_month, new_month):

    old_img = find_file(aoi, old_month, ".tif")
    new_img = find_file(aoi, new_month, ".tif")

    old_geo = find_file(
        aoi,
        old_month,
        "_Buildings.geojson"
    )

    new_geo = find_file(
        aoi,
        new_month,
        "_Buildings.geojson"
    )

    if not all([old_img, new_img, old_geo, new_geo]):
        print("  SKIP: missing file")
        return 0, 0, 0

    try:

        with rasterio.open(old_img) as src:
            old = src.read()

        with rasterio.open(new_img) as src:
            new = src.read()

    except Exception as e:

        print("  SKIP image error:", e)
        return 0, 0, 0

    if old.shape != new.shape:
        print("  SKIP: image shapes differ")
        return 0, 0, 0

    # Read annotations
    try:
        g_old = gpd.read_file(old_geo)
        g_new = gpd.read_file(new_geo)
    except Exception as e:
        print("  SKIP geojson error:", e)
        return 0, 0, 0

    if (
        len(g_old) == 0
        or len(g_new) == 0
        or "Id" not in g_old.columns
        or "Id" not in g_new.columns
    ):
        print("  SKIP: empty/missing Id")
        return 0, 0, 0

    old_ids = set(g_old["Id"])
    new_buildings = g_new[
        ~g_new["Id"].isin(old_ids)
    ]

    height = old.shape[1]
    width = old.shape[2]

    # Geometry is already in pixel coordinates
    transform = Affine.identity()

    if len(new_buildings) > 0:

        mask = rasterize(
            [
                (geom, 1)
                for geom in new_buildings.geometry
                if geom is not None
            ],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

    else:

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

    pair_name = f"{old_month}_to_{new_month}"

    output_dir = os.path.join(
        OUTPUT_ROOT,
        aoi,
        pair_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    patch_count = 0
    change_patches = 0
    changed_pixels = 0

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            PATCH_SIZE
        ):

            old_patch = old[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            new_patch = new[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            mask_patch = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            idx = patch_count

            np.save(
                os.path.join(
                    output_dir,
                    f"{idx}_old.npy"
                ),
                old_patch
            )

            np.save(
                os.path.join(
                    output_dir,
                    f"{idx}_new.npy"
                ),
                new_patch
            )

            np.save(
                os.path.join(
                    output_dir,
                    f"{idx}_mask.npy"
                ),
                mask_patch
            )

            patch_count += 1

            pixels = int(
                np.sum(mask_patch > 0)
            )

            changed_pixels += pixels

            if pixels > 0:
                change_patches += 1

    return (
        patch_count,
        change_patches,
        changed_pixels
    )


# ============================================================
# MAIN
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

aois = sorted(
    [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(
            os.path.join(DATA_ROOT, d)
        )
    ]
)

print("AOIs found:", len(aois))

total_patches = 0
total_change_patches = 0
total_changed_pixels = 0
total_pixels = 0

print()

for aoi in aois:

    start_month = get_start_month(aoi)

    print("=" * 70)
    print(aoi)

    print(
        f"  Processing "
        f"{start_month} -> 2019_12"
    )

    patches, change_patches, changed_pixels = create_patches(
        aoi,
        start_month,
        "2019_12"
    )

    print(
        f"    Patches: {patches} | "
        f"Change patches: {change_patches} | "
        f"Changed pixels: {changed_pixels}"
    )

    total_patches += patches
    total_change_patches += change_patches
    total_changed_pixels += changed_pixels

    total_pixels += patches * PATCH_SIZE * PATCH_SIZE


print()
print("=" * 70)
print("BEST INTERVAL PATCH CREATION COMPLETE")
print("=" * 70)

print("Total patches:", total_patches)
print("Patches with change:", total_change_patches)
print("Changed pixels:", total_changed_pixels)
print("Total pixels:", total_pixels)

if total_patches > 0:

    print(
        "Change-patch ratio:",
        f"{100 * total_change_patches / total_patches:.2f}%"
    )

    print(
        "Pixel change ratio:",
        f"{100 * total_changed_pixels / total_pixels:.4f}%"
    )
