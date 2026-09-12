import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_temporal import TemporalChangeDataset
from model_temporal import UNet
from dataset_temporal import TemporalChangeDataset, make_weighted_sampler


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)


# --------------------------------------------------
# AOI split
# --------------------------------------------------

ROOT = "temporal_patches"

AOIS = sorted([
    aoi for aoi in __import__("os").listdir(ROOT)
    if __import__("os").path.isdir(
        __import__("os").path.join(ROOT, aoi)
    )
])

train_aois = AOIS[:8]
val_aois = AOIS[8:]

print("Training AOIs:", len(train_aois))
print("Validation AOIs:", len(val_aois))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = TemporalChangeDataset(
    ROOT,
    train_aois,
    augment=False
)

val_dataset = TemporalChangeDataset(
    ROOT,
    val_aois,
    augment=False
)

print("Training patches:", len(train_dataset))
print("Validation patches:", len(val_dataset))


# --------------------------------------------------
# Data loaders
# --------------------------------------------------

sampler = make_weighted_sampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet(in_channels=8).to(device)


# --------------------------------------------------
# Losses
# --------------------------------------------------

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1e-6

    intersection = (pred * target).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(pred, target):

    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    dice = (
        2 * intersection
    ) / (
        pred.sum() + target.sum() + 1e-6
    )

    iou = (
        intersection
    ) / (
        union + 1e-6
    )

    return dice.item(), iou.item()


# --------------------------------------------------
# Training
# --------------------------------------------------

epochs = 25
best_iou = 0.0

for epoch in range(epochs):

    # ---------------------------
    # Train
    # ---------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            bce_loss(outputs, masks)
            + dice_loss(outputs, masks)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ---------------------------
    # Validation
    # ---------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = (
                bce_loss(outputs, masks)
                + dice_loss(outputs, masks)
            )

            val_loss += loss.item()

            dice, iou = calculate_metrics(
                outputs,
                masks
            )

            val_dice += dice
            val_iou += iou

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)


    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )


    # ---------------------------
    # Save best model
    # ---------------------------

    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(
            model.state_dict(),
            "checkpoints/best_model_temporal.pth"
        )

        print(
            f"  ✓ Saved best model "
            f"(IoU: {best_iou:.4f})"
        )


print("\nTraining complete.")
print(f"Best validation IoU: {best_iou:.4f}")
