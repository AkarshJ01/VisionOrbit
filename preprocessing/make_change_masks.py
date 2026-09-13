import os
import sys
from pathlib import Path
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from PIL import Image
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = os.path.join(str(PROJECT_ROOT), "data")

for aoi_dir in sorted(glob.glob(os.path.join(DATA_DIR, "L15-*"))):

    if not os.path.isdir(aoi_dir):
        continue

    aoi = os.path.basename(aoi_dir)

    # Find the two label files
    label_files = sorted(glob.glob(os.path.join(aoi_dir, "*_Buildings.geojson")))

    if len(label_files) != 2:
        print(f"SKIPPING {aoi}: expected 2 label files, found {len(label_files)}")
        continue

    old_label = [f for f in label_files if "2018_01" in f or "2018_02" in f][0]
    new_label = [f for f in label_files if "2019_12" in f][0]

    # Find imagery
    old_image = [f for f in glob.glob(os.path.join(aoi_dir, "*.tif"))
                 if "2018_01" in f or "2018_02" in f][0]

    new_image = [f for f in glob.glob(os.path.join(aoi_dir, "*.tif"))
                 if "2019_12" in f][0]

    print(f"\nProcessing: {aoi}")
    print(f"  Old: {os.path.basename(old_image)}")
    print(f"  New: {os.path.basename(new_image)}")

    # Read building labels
    g18 = gpd.read_file(old_label)
    g19 = gpd.read_file(new_label)

    # Persistent building IDs
    ids18 = set(g18["Id"])
    ids19 = set(g19["Id"])

    new_ids = ids19 - ids18
    missing_ids = ids18 - ids19

    new_buildings = g19[g19["Id"].isin(new_ids)]

    # Get image dimensions
    with rasterio.open(new_image) as src:
        height = src.height
        width = src.width

    # IMPORTANT:
    # labels_match_pix contains pixel-coordinate geometries.
    # Therefore we use an identity transform.
    shapes = [(geom, 1) for geom in new_buildings.geometry if geom is not None]

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=Affine.identity(),
        fill=0,
        dtype=np.uint8
    )

    # Save binary mask
    mask_path = os.path.join(aoi_dir, "change_mask.png")
    Image.fromarray(mask * 255).save(mask_path)

    # Also save NumPy version for training later
    np.save(os.path.join(aoi_dir, "change_mask.npy"), mask)

    print(f"  2018 buildings: {len(ids18)}")
    print(f"  2019 buildings: {len(ids19)}")
    print(f"  NEW buildings:  {len(new_ids)}")
    print(f"  Missing:        {len(missing_ids)}")
    print(f"  Changed pixels: {int(mask.sum())}")
    print(f"  Saved: {mask_path}")

print("\nDONE.")
