import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MultiSensorDataset, resolve_data_root, IMAGE_DIR, MASK_DIR
from model import UNet


# --------------------------------------------------
# Loss Functions & Metrics
# --------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and DiceLoss.
    Combines pixel-level classification with structural overlap.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MetricTracker:
    """
    Accumulates true positives, false positives, and false negatives
    across an entire epoch to compute exact dataset-level IoU and Dice.
    """
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.total_tp = 0.0
        self.total_fp = 0.0
        self.total_fn = 0.0
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, logits, targets, loss=0.0):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)

        tp = (preds_flat * targets_flat).sum().item()
        fp = (preds_flat * (1.0 - targets_flat)).sum().item()
        fn = ((1.0 - preds_flat) * targets_flat).sum().item()

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_loss += loss
        self.num_batches += 1

    def compute(self):
        iou = (self.total_tp + self.smooth) / (self.total_tp + self.total_fp + self.total_fn + self.smooth)
        dice = (2.0 * self.total_tp + self.smooth) / (2.0 * self.total_tp + self.total_fp + self.total_fn + self.smooth)
        avg_loss = self.total_loss / max(1, self.num_batches)
        return avg_loss, iou, dice


# --------------------------------------------------
# Data Augmentations (In-place Tensor Transformations)
# --------------------------------------------------

def augment_batch(images, masks):
    """
    Spatial augmentations for nadir satellite imagery:
    Random horizontal flip, vertical flip, and 90-degree rotations.
    """
    # Random Horizontal Flip
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[-1])
        masks = torch.flip(masks, dims=[-1])

    # Random Vertical Flip
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[-2])
        masks = torch.flip(masks, dims=[-2])

    # Random 90-degree Rotation
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        images = torch.rot90(images, k=k, dims=[-2, -1])
        masks = torch.rot90(masks, k=k, dims=[-2, -1])

    return images.contiguous(), masks.contiguous()


# --------------------------------------------------
# Train and Validation Loops
# --------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, augment=True):
    model.train()
    tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if augment:
            images, masks = augment_batch(images, masks)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        tracker.update(logits, masks, loss.item())

    return tracker.compute()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        tracker.update(logits, masks, loss.item())

    return tracker.compute()


# --------------------------------------------------
# Main Training Function
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Sensor U-Net for Building Footprint Extraction")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--no-augment", action="store_true", help="Disable spatial data augmentation")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory (default: auto-detected)")
    args = parser.parse_args()

    # Hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_root = resolve_data_root(args.data_dir)
    image_dir = data_root / "processed" / "images" if (data_root / "processed" / "images").exists() else data_root / "images"
    mask_dir = data_root / "processed" / "masks" if (data_root / "processed" / "masks").exists() else data_root / "masks"

    print("=" * 65)
    print("VISIONORBIT: MULTI-SENSOR U-NET TRAINING")
    print("=" * 65)
    print(f"Device:            {device}")
    print(f"Data Root:         {data_root.resolve()}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Initial LR:        {args.lr}")
    print(f"Data Augmentation: {not args.no_augment}")
    print("=" * 65)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Tile splits
    train_tiles = [55, 69, 783, 8137, 4164, 108, 442]
    val_tiles = [7924, 2317]

    train_dataset = MultiSensorDataset(image_dir, mask_dir, train_tiles)
    val_dataset = MultiSensorDataset(image_dir, mask_dir, val_tiles)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nTraining Samples:   {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Validation Samples: {len(val_dataset)} ({len(val_loader)} batches)")

    # Model, Criterion, Optimizer, Scheduler
    model = UNet(in_channels=7, out_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_iou = -1.0
    history = {
        "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss": [], "val_iou": [], "val_dice": [], "lr": []
    }

    print("\nStarting Training...\n")
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Train IoU':<9} | {'Val Loss':<9} | {'Val IoU':<8} | {'Val Dice':<8} | {'LR':<8} | {'Time':<5}")
    print("-" * 75)

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, augment=(not args.no_augment)
        )

        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        print(
            f"{epoch:<7d} | {train_loss:<10.4f} | {train_iou:<9.4f} | "
            f"{val_loss:<9.4f} | {val_iou:<8.4f} | {val_dice:<8.4f} | {current_lr:<8.2e} | {epoch_duration:<4.1f}s",
            end=""
        )

        # Checkpoint if best validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
                "val_loss": val_loss,
            }
            torch.save(best_checkpoint, save_path / "best_model.pth")
            print("  <-- [Saved Best]")
        else:
            print()

        # Save latest checkpoint
        latest_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_iou,
            "val_dice": val_dice,
            "val_loss": val_loss,
        }
        torch.save(latest_checkpoint, save_path / "latest_model.pth")

    total_time = time.time() - start_total_time
    print("-" * 75)
    print(f"\nTraining Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best model saved to: {save_path / 'best_model.pth'}")

    # Save history JSON
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
