import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = os.path.join(str(PROJECT_ROOT), "data")
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "data/patches/temporal")
PATCH_SIZE = 256

AOIS = [
    "L15-0331E-1257N_1327_3160_13",
    "L15-0357E-1223N_1429_3296_13",
    "L15-0358E-1220N_1433_3310_13",
    "L15-0361E-1300N_1446_2989_13",
    "L15-0368E-1245N_1474_3210_13",
    "L15-0387E-1276N_1549_3087_13",
    "L15-0434E-1218N_1736_3318_13",
    "L15-0457E-1135N_1831_3648_13",
    "L15-0487E-1246N_1950_3207_13",
    "L15-0506E-1204N_2027_3374_13",
]

PAIRS = [
    ("2018_01", "2018_02"),
    ("2018_06", "2018_07"),
    ("2019_01", "2019_02"),
    ("2019_06", "2019_07"),
    ("2019_11", "2019_12"),
]


def get_image_path(aoi, month):
    folder = os.path.join(DATA_DIR, aoi)

    # Original files use shorter names
    if month == "2018_01":
        return os.path.join(folder, "2018_01.tif")
    if month == "2019_12":
        return os.path.join(folder, "2019_12.tif")
    if month == "2018_02" and aoi == "L15-0358E-1220N_1433_3310_13":
        return os.path.join(
            folder,
            f"global_monthly_2018_02_mosaic_{aoi}.tif"
        )

    return os.path.join(
        folder,
        f"global_monthly_{month}_mosaic_{aoi}.tif"
    )


def get_label_path(aoi, month):
    folder = os.path.join(DATA_DIR, aoi)

    if month == "2018_01":
        return os.path.join(folder, "2018_01_Buildings.geojson")
    if month == "2019_12":
        return os.path.join(folder, "2019_12_Buildings.geojson")
    if month == "2018_02" and aoi == "L15-0358E-1220N_1433_3310_13":
        return os.path.join(
            folder,
            f"global_monthly_2018_02_mosaic_{aoi}_Buildings.geojson"
        )

    return os.path.join(
        folder,
        f"global_monthly_{month}_mosaic_{aoi}_Buildings.geojson"
    )


def create_pair(aoi, old_month, new_month):

    old_image = get_image_path(aoi, old_month)
    new_image = get_image_path(aoi, new_month)
    old_labels = get_label_path(aoi, old_month)
    new_labels = get_label_path(aoi, new_month)

    # Skip unavailable pairs
    required = [old_image, new_image, old_labels, new_labels]

    if not all(os.path.exists(x) for x in required):
        print(f"SKIP {aoi}: {old_month} -> {new_month} (missing file)")
        return 0

    print(f"\nProcessing {aoi}: {old_month} -> {new_month}")

    # Read imagery
    with rasterio.open(old_image) as src:
        old_img = src.read()
        height, width = src.height, src.width

    with rasterio.open(new_image) as src:
        new_img = src.read()

    # Read building labels
    old_gdf = gpd.read_file(old_labels)
    new_gdf = gpd.read_file(new_labels)

    # Some SpaceNet files can contain no annotations.
    # Do not treat an empty annotation file as "no change".
    if len(old_gdf) == 0 or len(new_gdf) == 0:
        print("  SKIP: one of the annotation files is empty")
        return 0

    if "Id" not in old_gdf.columns or "Id" not in new_gdf.columns:
        print("  SKIP: annotation file has no Id column")
        return 0

    old_ids = set(old_gdf["Id"])
    new_ids = set(new_gdf["Id"])

    # Buildings appearing in the new month
    new_building_ids = new_ids - old_ids
    new_buildings = new_gdf[
        new_gdf["Id"].isin(new_building_ids)
    ]

    print(f"  Old buildings: {len(old_ids)}")
    print(f"  New buildings: {len(new_ids)}")
    print(f"  New building IDs: {len(new_building_ids)}")

    # GeoJSON coordinates are already pixel coordinates
    shapes = [
        (geom, 1)
        for geom in new_buildings.geometry
        if geom is not None and not geom.is_empty
    ]

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=rasterio.Affine.identity(),
        fill=0,
        dtype=np.uint8
    )

    pair_name = f"{old_month}_to_{new_month}"
    output_dir = os.path.join(
        OUTPUT_DIR,
        aoi,
        pair_name
    )

    os.makedirs(output_dir, exist_ok=True)

    patch_count = 0
    changed_patches = 0

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

            old_patch = old_img[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            new_patch = new_img[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            mask_patch = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Make sure patch is exactly the right size
            if (
                old_patch.shape[1:] != (PATCH_SIZE, PATCH_SIZE)
                or new_patch.shape[1:] != (PATCH_SIZE, PATCH_SIZE)
            ):
                continue

            np.save(
                os.path.join(output_dir, f"{patch_count}_old.npy"),
                old_patch
            )

            np.save(
                os.path.join(output_dir, f"{patch_count}_new.npy"),
                new_patch
            )

            np.save(
                os.path.join(output_dir, f"{patch_count}_mask.npy"),
                mask_patch
            )

            if mask_patch.sum() > 0:
                changed_patches += 1

            patch_count += 1

    print(f"  Patches: {patch_count}")
    print(f"  Patches with change: {changed_patches}")

    return patch_count


def main():

    total_patches = 0
    total_pairs = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for aoi in AOIS:

        for old_month, new_month in PAIRS:

            # AOI 3 starts at February 2018
            if (
                aoi == "L15-0358E-1220N_1433_3310_13"
                and old_month == "2018_01"
            ):
                print(
                    f"SKIP {aoi}: no 2018_01 imagery"
                )
                continue

            patches = create_pair(
                aoi,
                old_month,
                new_month
            )

            if patches > 0:
                total_patches += patches
                total_pairs += 1

    print("\n" + "=" * 60)
    print("TEMPORAL PATCH CREATION COMPLETE")
    print("=" * 60)
    print(f"Temporal pairs processed: {total_pairs}")
    print(f"Total patches created: {total_patches}")


if __name__ == "__main__":
    main()
