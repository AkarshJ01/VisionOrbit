import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class LongIntervalChangeDataset(Dataset):

    def __init__(self, root, aois, augment=False):
        self.root = root
        self.aois = set(aois)
        self.augment = augment

        self.samples = []

        # Find all temporal pairs
        for aoi in sorted(self.aois):

            aoi_path = os.path.join(root, aoi)

            if not os.path.isdir(aoi_path):
                continue

            pair_dirs = sorted(
                [
                    d for d in os.listdir(aoi_path)
                    if os.path.isdir(os.path.join(aoi_path, d))
                ]
            )

            for pair in pair_dirs:

                pair_path = os.path.join(aoi_path, pair)

                old_files = sorted(
                    glob.glob(os.path.join(pair_path, "*_old.npy"))
                )

                for old_file in old_files:

                    new_file = old_file.replace("_old.npy", "_new.npy")
                    mask_file = old_file.replace("_old.npy", "_mask.npy")

                    if (
                        os.path.exists(new_file)
                        and os.path.exists(mask_file)
                    ):
                        self.samples.append(
                            (old_file, new_file, mask_file)
                        )

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        old_file, new_file, mask_file = self.samples[idx]

        # Load images
        old = np.load(old_file).astype(np.float32) / 255.0
        new = np.load(new_file).astype(np.float32) / 255.0

        # Load change mask
        mask = np.load(mask_file).astype(np.float32)

        # --------------------------------------------------
        # Input:
        # [old 4 bands, new 4 bands]
        # Shape = (8, 256, 256)
        # --------------------------------------------------

        image = np.concatenate([old, new], axis=0)

        # --------------------------------------------------
        # Optional augmentation
        # Disabled initially for our experiment
        # --------------------------------------------------

        if self.augment:

            # Horizontal flip
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            # Vertical flip
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            # Random 90-degree rotation
            k = np.random.randint(0, 4)

            if k > 0:
                image = np.rot90(
                    image,
                    k=k,
                    axes=(1, 2)
                ).copy()

                mask = np.rot90(
                    mask,
                    k=k,
                    axes=(0, 1)
                ).copy()

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# ============================================================
# VERIFY DATASET
# ============================================================

if __name__ == "__main__":

    ROOT = "long_interval_patches"

    # Get AOIs
    aois = sorted(
        [
            d for d in os.listdir(ROOT)
            if os.path.isdir(os.path.join(ROOT, d))
        ]
    )

    print("\nAOIs:")
    for aoi in aois:
        print(" ", aoi)

    # Same geographic split as before
    train_aois = aois[:8]
    val_aois = aois[8:]

    print("\nTraining AOIs:")
    for aoi in train_aois:
        print(" ", aoi)

    print("\nValidation AOIs:")
    for aoi in val_aois:
        print(" ", aoi)

    # No augmentation for the first experiment
    train_dataset = LongIntervalChangeDataset(
        ROOT,
        train_aois,
        augment=False
    )

    val_dataset = LongIntervalChangeDataset(
        ROOT,
        val_aois,
        augment=False
    )

    print("\n========================================")
    print("DATASET VERIFICATION")
    print("========================================")

    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print("Total patches:", len(train_dataset) + len(val_dataset))

    # Inspect one sample
    image, mask = train_dataset[0]

    print("\nSample:")
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Image range:", image.min().item(), "to", image.max().item())
    print("Mask values:", torch.unique(mask))

    # Count positive pixels
    total_pixels = 0
    changed_pixels = 0

    for _, _, mask_file in train_dataset.samples + val_dataset.samples:

        mask = np.load(mask_file)

        total_pixels += mask.size
        changed_pixels += np.sum(mask > 0)

    print("\nChange statistics:")
    print("Total pixels:", total_pixels)
    print("Changed pixels:", changed_pixels)

    if total_pixels > 0:
        print(
            "Pixel change ratio:",
            f"{100 * changed_pixels / total_pixels:.4f}%"
        )

    print("\nDataset verification complete.")
