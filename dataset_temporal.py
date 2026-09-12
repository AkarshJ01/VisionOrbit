import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler


class TemporalChangeDataset(Dataset):

    def __init__(self, root_dir, aois, augment=False):

        self.root_dir = root_dir
        self.aois = set(aois)
        self.augment = augment

        self.samples = []

        for aoi in sorted(os.listdir(root_dir)):

            if aoi not in self.aois:
                continue

            aoi_dir = os.path.join(root_dir, aoi)

            if not os.path.isdir(aoi_dir):
                continue

            for pair in sorted(os.listdir(aoi_dir)):

                pair_dir = os.path.join(aoi_dir, pair)

                if not os.path.isdir(pair_dir):
                    continue

                old_files = sorted(
                    glob.glob(os.path.join(pair_dir, "*_old.npy"))
                )

                for old_path in old_files:

                    filename = os.path.basename(old_path)
                    index = filename.replace("_old.npy", "")

                    new_path = os.path.join(
                        pair_dir,
                        f"{index}_new.npy"
                    )

                    mask_path = os.path.join(
                        pair_dir,
                        f"{index}_mask.npy"
                    )

                    if (
                        os.path.exists(new_path)
                        and os.path.exists(mask_path)
                    ):
                        self.samples.append(
                            (
                                old_path,
                                new_path,
                                mask_path
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        old_path, new_path, mask_path = self.samples[idx]

        old = np.load(old_path).astype(np.float32) / 255.0
        new = np.load(new_path).astype(np.float32) / 255.0
        mask = np.load(mask_path).astype(np.float32)

        # 4 old bands + 4 new bands
        image = np.concatenate(
            [old, new],
            axis=0
        )

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Training augmentation
        if self.augment:

            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])

            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])

            rotations = torch.randint(0, 4, (1,)).item()

            image = torch.rot90(
                image,
                rotations,
                dims=[1, 2]
            )

            mask = torch.rot90(
                mask,
                rotations,
                dims=[1, 2]
            )

        return image, mask


if __name__ == "__main__":

    root = "temporal_patches"

    aois = sorted(os.listdir(root))

    train_aois = aois[:8]
    val_aois = aois[8:]

    print("Training AOIs:")
    for aoi in train_aois:
        print(" ", aoi)

    print("\nValidation AOIs:")
    for aoi in val_aois:
        print(" ", aoi)

    train_dataset = TemporalChangeDataset(
        root,
        train_aois,
        augment=True
    )

    val_dataset = TemporalChangeDataset(
        root,
        val_aois,
        augment=False
    )

    print("\nTraining patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    image, mask = train_dataset[0]

    print("\nSample:")
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Image range:", image.min().item(), "to", image.max().item())
    print("Mask values:", torch.unique(mask))

def make_weighted_sampler(dataset):
    weights = []

    for i in range(len(dataset)):
        mask = dataset[i][1]

        # Does this patch contain any change?
        has_change = mask.sum().item() > 0

        # Give change patches much higher probability
        if has_change:
            weights.append(4.0)
        else:
            weights.append(1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

