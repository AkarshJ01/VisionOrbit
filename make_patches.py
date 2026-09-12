import os
import glob
import numpy as np
import rasterio

PATCH_SIZE = 256

os.makedirs("patches", exist_ok=True)

for aoi_dir in sorted(glob.glob("data/*")):

    if not os.path.isdir(aoi_dir):
        continue

    aoi = os.path.basename(aoi_dir)

    # Find the two images
    old_files = glob.glob(os.path.join(aoi_dir, "2018_*.tif"))
    new_files = glob.glob(os.path.join(aoi_dir, "2019_12.tif"))

    if not old_files or not new_files:
        print(f"Skipping {aoi}: images not found")
        continue

    old_path = old_files[0]
    new_path = new_files[0]
    mask_path = os.path.join(aoi_dir, "change_mask.npy")

    if not os.path.exists(mask_path):
        print(f"Skipping {aoi}: change mask not found")
        continue

    # Read images
    with rasterio.open(old_path) as src:
        old_img = src.read()

    with rasterio.open(new_path) as src:
        new_img = src.read()

    # Read change mask
    mask = np.load(mask_path)

    # Make sure everything has matching dimensions
    height = min(old_img.shape[1], new_img.shape[1], mask.shape[0])
    width = min(old_img.shape[2], new_img.shape[2], mask.shape[1])

    old_img = old_img[:, :height, :width]
    new_img = new_img[:, :height, :width]
    mask = mask[:height, :width]

    # Output folder
    out_dir = os.path.join("patches", aoi)
    os.makedirs(out_dir, exist_ok=True)

    patch_number = 0

    # Split into 256x256 patches
    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

            old_patch = old_img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            new_patch = new_img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Save input images and target separately
            np.save(
                os.path.join(out_dir, f"{patch_number}_old.npy"),
                old_patch
            )

            np.save(
                os.path.join(out_dir, f"{patch_number}_new.npy"),
                new_patch
            )

            np.save(
                os.path.join(out_dir, f"{patch_number}_mask.npy"),
                mask_patch
            )

            patch_number += 1

    print(
        f"{aoi}: created {patch_number} patches "
        f"({old_img.shape[1]}x{old_img.shape[2]})"
    )

print("\nDone.")
