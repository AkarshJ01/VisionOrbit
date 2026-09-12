import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MultiSensorDataset, IMAGE_DIR, MASK_DIR
from model import UNet


# --------------------------------------------------
# Loss Functions & Metrics for Testing
# --------------------------------------------------

class DiceLoss(nn.Module):
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


def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_preds = preds_flat.sum().item()
    total_targets = targets_flat.sum().item()

    union = total_preds + total_targets - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_preds + total_targets + smooth)

    return iou, dice


# --------------------------------------------------
# Test Pipeline
# --------------------------------------------------

def test_pipeline():
    print("=" * 60)
    print("RUNNING END-TO-END MODEL INTEGRATION TEST")
    print("=" * 60)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[1] Target Device: {device}")

    # 2. Load batch (real if dataset exists, synthetic fallback if not yet downloaded)
    try:
        dataset = MultiSensorDataset(IMAGE_DIR, MASK_DIR)
        if len(dataset) == 0:
            raise ValueError("No dataset samples found")
        batch_size = min(2, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        images, masks = next(iter(loader))
        print(f"[2] Sample Multi-Sensor Batch Loaded:")
        print(f"    Images: shape={images.shape}, dtype={images.dtype}, min={images.min():.3f}, max={images.max():.3f}")
        print(f"    Masks:  shape={masks.shape}, dtype={masks.dtype}, unique={torch.unique(masks).tolist()}")
    except (FileNotFoundError, IndexError, ValueError):
        print(f"[2] [NOTICE] Full dataset not found on this path.")
        print(f"    Generating synthetic 7-channel multi-sensor batch for verification...")
        images = torch.rand(2, 7, 256, 256, dtype=torch.float32)
        masks = (torch.rand(2, 1, 256, 256) > 0.85).float()
        print(f"    Synthetic batch generated: images={images.shape}, masks={masks.shape}")

    # 3. Model instantiation & forward pass
    model = UNet(in_channels=7, out_channels=1).to(device)
    images = images.to(device)
    masks = masks.to(device)

    print(f"[3] Model instantiated and moved to {device}")
    logits = model(images)
    print(f"    Forward pass successful! Output logits shape: {logits.shape}")

    # 4. Loss computation
    criterion = CombinedLoss()
    loss = criterion(logits, masks)
    print(f"[4] Loss computation successful!")
    print(f"    Combined Loss: {loss.item():.4f}")

    # 5. Backward pass & optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[5] Backward pass and optimizer step successful!")

    # 6. Evaluation metrics
    iou, dice = calculate_metrics(logits, masks)
    print(f"[6] Metric calculation successful!")
    print(f"    Batch IoU:  {iou:.4f}")
    print(f"    Batch Dice: {dice:.4f}")

    print("=" * 60)
    print("ALL TESTS PASSED! Ready for training or inference.")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
