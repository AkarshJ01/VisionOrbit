from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "data" / "SpaceNet6"

IMAGE_DIR = DATA_ROOT / "processed" / "images"
MASK_DIR = DATA_ROOT / "processed" / "masks"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SpaceNetDataset(Dataset):

    def __init__(self, image_dir, mask_dir, tile_ids):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Store tile IDs as strings
        self.tile_ids = set(str(tile) for tile in tile_ids)

        self.samples = []

        # Find all .npy image files
        for filename in sorted(self.image_dir.glob("*.npy")):

            # Example filename:
            # tile_108_y0_x0.npy

            parts = filename.stem.split("_")

            # parts = ["tile", "108", "y0", "x0"]
            tile_id = parts[1]

            if tile_id in self.tile_ids:

                mask_path = self.mask_dir / filename.name

                if mask_path.exists():
                    self.samples.append(
                        (filename, mask_path)
                    )

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