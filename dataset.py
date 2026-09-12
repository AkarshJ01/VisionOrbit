from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Paths & Dynamic Data Discovery
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_data_root(custom_path=None):
    """
    Dynamically finds the dataset directory:
    1. Custom path if explicitly provided
    2. VISIONORBIT_DATA_DIR environment variable
    3. VisionOrbit/data/
    4. VisionOrbit/demo_data/
    5. Parent workspace data directory
    """
    if custom_path:
        p = Path(custom_path)
        if (p / "processed" / "images").exists():
            return p
        elif (p / "images").exists():
            return p
        return p

    env_dir = os.environ.get("VISIONORBIT_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # Check local repo data directory
    local_data = PROJECT_ROOT / "data"
    if (local_data / "processed" / "images").exists():
        return local_data
    if (local_data / "images").exists():
        return local_data

    # Check parent workspace data directory
    parent_data = PROJECT_ROOT.parent / "data"
    if parent_data.exists():
        for sub in parent_data.iterdir():
            if sub.is_dir() and (sub / "processed" / "images").exists():
                return sub
        if (parent_data / "processed" / "images").exists():
            return parent_data

    # Check demo data directory
    demo_dir = PROJECT_ROOT / "demo_data"
    if demo_dir.exists():
        return demo_dir

    return local_data


DATA_ROOT = resolve_data_root()
if (DATA_ROOT / "processed" / "images").exists():
    IMAGE_DIR = DATA_ROOT / "processed" / "images"
    MASK_DIR = DATA_ROOT / "processed" / "masks"
elif (DATA_ROOT / "images").exists():
    IMAGE_DIR = DATA_ROOT / "images"
    MASK_DIR = DATA_ROOT / "masks"
else:
    IMAGE_DIR = DATA_ROOT
    MASK_DIR = DATA_ROOT


# --------------------------------------------------
# Multi-Sensor Dataset
# --------------------------------------------------

class MultiSensorDataset(Dataset):
    """
    Dataset loader for 7-channel multimodal satellite patches:
    - 3 Optical RGB channels
    - 4 Synthetic Aperture Radar (SAR) polarimetric channels
    - 1 Binary building footprint segmentation mask
    """

    def __init__(self, image_dir=None, mask_dir=None, tile_ids=None):

        if image_dir is None:
            image_dir = IMAGE_DIR
        if mask_dir is None:
            mask_dir = MASK_DIR

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Multi-sensor dataset directory not found at: '{self.image_dir.resolve()}'\n"
                f"How to resolve this:\n"
                f"1. Specify path via --data-dir or set VISIONORBIT_DATA_DIR environment variable.\n"
                f"2. Use the included demo_data/ folder for quick testing.\n"
                f"3. See README.md for instructions on training and inference."
            )

        # Store tile IDs as strings (None means load all tiles found)
        self.tile_ids = set(str(tile) for tile in tile_ids) if tile_ids is not None else None

        self.samples = []

        # Find all .npy image files
        for filename in sorted(self.image_dir.glob("*.npy")):
            # Skip mask files if image_dir == mask_dir
            if "mask" in filename.name:
                continue

            parts = filename.stem.split("_")
            tile_id = parts[1] if len(parts) > 1 else "default"

            if self.tile_ids is None or tile_id in self.tile_ids:
                mask_path = self.mask_dir / filename.name
                if not mask_path.exists():
                    alt_mask = self.mask_dir / f"{filename.stem}_mask.npy"
                    if alt_mask.exists():
                        mask_path = alt_mask

                if mask_path.exists():
                    self.samples.append((filename, mask_path))
                else:
                    self.samples.append((filename, None))

        if self.tile_ids:
            print(f"Tiles: {sorted(self.tile_ids)}")
        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = np.load(image_path).astype(np.float32)
        image = torch.from_numpy(image)

        if mask_path and Path(mask_path).exists():
            mask = np.load(mask_path).astype(np.float32)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)

        return image, mask


# --------------------------------------------------
# Test the dataset
# --------------------------------------------------

if __name__ == "__main__":
    train_tiles = [55, 69, 783, 8137, 4164, 108, 442]
    val_tiles = [7924, 2317]
    test_tiles = [7218]

    train_dataset = MultiSensorDataset(IMAGE_DIR, MASK_DIR, train_tiles)
    val_dataset = MultiSensorDataset(IMAGE_DIR, MASK_DIR, val_tiles)
    test_dataset = MultiSensorDataset(IMAGE_DIR, MASK_DIR, test_tiles)

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(train_loader))

    print("\nFirst batch:")
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Image range:", images.min().item(), "to", images.max().item())
    print("Mask values:", torch.unique(masks).tolist())