import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceNetChangeDataset(Dataset):

    def __init__(self, patches_dir=None, split="train", augment=False):
        if patches_dir is None:
            patches_dir = "data/patches/base" if os.path.exists("data/patches/base") else "patches"
        self.patches_dir = patches_dir
        self.split = split
        self.augment = augment

        all_patches = sorted(
            glob.glob(os.path.join(patches_dir, "*", "*_old.npy"))
        )

        # Get AOI names
        aois = sorted(
            set(os.path.basename(os.path.dirname(p)) for p in all_patches)
        )

        # Same split as before
        train_aois = aois[:8]
        val_aois = aois[8:]

        if split == "train":
            selected_aois = train_aois
        else:
            selected_aois = val_aois

        self.files = [
            p for p in all_patches
            if os.path.basename(os.path.dirname(p)) in selected_aois
        ]

        print(f"Total patches: {len(self.files)}")
        print(f"Total AOIs: {len(selected_aois)}")

        if split == "train":
            print("Training AOIs:")
        else:
            print("Validation AOIs:")

        for aoi in selected_aois:
            print(" ", aoi)

        print(f"{split.capitalize()} patches: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        old_path = self.files[idx]
        new_path = old_path.replace("_old.npy", "_new.npy")
        mask_path = old_path.replace("_old.npy", "_mask.npy")

        old = np.load(old_path).astype(np.float32) / 255.0
        new = np.load(new_path).astype(np.float32) / 255.0
        mask = np.load(mask_path).astype(np.float32)

        # Explicit change information
        difference = np.abs(new - old)

        # 4 old + 4 new + 4 difference = 12 channels
        image = np.concatenate(
            [old, new, difference],
            axis=0
        )

        # Data augmentation — training only
        if self.augment:

            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            # Random 90-degree rotation
            k = np.random.randint(0, 4)

            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


if __name__ == "__main__":

    dataset = SpaceNetChangeDataset(
        patches_dir="patches",
        split="train",
        augment=True
    )

    image, mask = dataset[0]

    print("\nSample:")
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Image range:", image.min().item(), "to", image.max().item())
    print("Mask values:", torch.unique(mask))
