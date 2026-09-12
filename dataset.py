from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


import os

# --------------------------------------------------
# Paths & Dynamic Data Discovery
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_data_root(custom_path=None):
    """
    Dynamically finds the SpaceNet6 data directory:
    1. Custom path if explicitly provided
    2. SPACENET_DATA_DIR environment variable
    3. VisionOrbit/data/SpaceNet6 (internal data directory)
    4. ../data/SpaceNet6 (parent workspace directory)
    """
    if custom_path:
        p = Path(custom_path)
        if (p / "processed" / "images").exists():
            return p
        elif (p / "images").exists():
            return p.parent
        return p

    env_dir = os.environ.get("SPACENET_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # Check local repo data
    local_data = PROJECT_ROOT / "data" / "SpaceNet6"
    if (local_data / "processed" / "images").exists():
        return local_data

    # Check parent workspace data
    parent_data = PROJECT_ROOT.parent / "data" / "SpaceNet6"
    if (parent_data / "processed" / "images").exists():
        return parent_data

    # Fallback default
    return parent_data


DATA_ROOT = resolve_data_root()
IMAGE_DIR = DATA_ROOT / "processed" / "images"
MASK_DIR = DATA_ROOT / "processed" / "masks"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SpaceNetDataset(Dataset):

    def __init__(self, image_dir=None, mask_dir=None, tile_ids=None):

        if image_dir is None:
            image_dir = IMAGE_DIR
        if mask_dir is None:
            mask_dir = MASK_DIR

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] SpaceNet6 images directory not found at: '{self.image_dir.resolve()}'\n"
                f"How to resolve this:\n"
                f"1. If data is stored elsewhere, pass image_dir or set SPACENET_DATA_DIR environment variable.\n"
                f"2. If you have raw SpaceNet 6 data, run 'python preprocess.py' from data/SpaceNet6/ to generate patches.\n"
                f"3. See README.md for instructions on downloading the SpaceNet 6 sample."
            )

        # Store tile IDs as strings (None means load all tiles found)
        self.tile_ids = set(str(tile) for tile in tile_ids) if tile_ids is not None else None

        self.samples = []

        # Find all .npy image files
        for filename in sorted(self.image_dir.glob("*.npy")):

            # Example filename:
            # tile_108_y0_x0.npy

            parts = filename.stem.split("_")

            # parts = ["tile", "108", "y0", "x0"]
            tile_id = parts[1] if len(parts) > 1 else "unknown"

            if self.tile_ids is None or tile_id in self.tile_ids:

                mask_path = self.mask_dir / filename.name

                if mask_path.exists():
                    self.samples.append(
                        (filename, mask_path)
                    )

        if self.tile_ids:
            print(f"Tiles: {sorted(self.tile_ids)}")
        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # Load NumPy arrays
        image = np.load(image_path)
        mask = np.load(mask_path)

        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # NumPy → PyTorch
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Mask:
        # (256, 256)
        #
        # Change to:
        # (1, 256, 256)

        mask = mask.unsqueeze(0)

        return image, mask


# --------------------------------------------------
# Test the dataset
# --------------------------------------------------

if __name__ == "__main__":

    # Original SpaceNet tiles
    train_tiles = [
        55,
        69,
        783,
        8137,
        4164,
        108,
        442
    ]

    val_tiles = [
        7924,
        2317
    ]

    test_tiles = [
        7218
    ]

    # Create datasets
    train_dataset = SpaceNetDataset(
        IMAGE_DIR,
        MASK_DIR,
        train_tiles
    )

    val_dataset = SpaceNetDataset(
        IMAGE_DIR,
        MASK_DIR,
        val_tiles
    )

    test_dataset = SpaceNetDataset(
        IMAGE_DIR,
        MASK_DIR,
        test_tiles
    )

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False
    )

    # --------------------------------------------------
    # Test one batch
    # --------------------------------------------------

    images, masks = next(iter(train_loader))

    print("\nFirst training batch:")
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

    print("\nImage dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)

    print("\nImage range:")
    print("Min:", images.min().item())
    print("Max:", images.max().item())

    print("\nMask values:", torch.unique(masks))