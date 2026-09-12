"""
Train RetinaNet (torchvision, ResNet50-FPN backbone) on FAIR1M.

Usage (from the `src/` folder):
    python train.py --data-root ../data --epochs 10 --batch-size 2

Notes:
- torchvision's RetinaNet does its own internal resizing/normalization
  (via GeneralizedRCNNTransform), so we feed it full-size images and it
  handles that internally -- no manual resize needed in the Dataset.
- Batch size is kept small by default because FAIR1M images are ~1024x1024
  and RetinaNet + ResNet50-FPN is memory hungry. Raise it if you have GPU
  headroom, lower it (or use gradient accumulation) if you hit OOM.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from model import get_model, get_device, get_param_groups
from dataset import FAIR1MDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data",
                    help="folder containing images/ and labelXmls/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="../models")
    p.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint .pth to resume from")
    return p.parse_args()


def evaluate_loss(model, data_loader, device):
    """RetinaNet only returns a loss dict in .train() mode, so we
    temporarily keep training mode on but disable gradient tracking to get
    a comparable validation loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    full_dataset = FAIR1MDataset(root=args.data_root)
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = get_model()
    model.to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}, continuing at epoch {start_epoch}")

    # Differential LR: pretrained backbone/FPN fine-tunes gently, the
    # freshly-initialized head (which starts heavily biased toward
    # ~1% confidence on every anchor by torchvision's focal-loss init)
    # gets a much larger LR so it actually moves in a realistic number
    # of epochs. See model.get_param_groups / model.py docstring.
    param_groups = get_param_groups(model, backbone_lr=args.lr * 0.1, head_lr=args.lr * 10)
    optimizer = torch.optim.SGD(
        param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if (i + 1) % 20 == 0:
                avg = running_loss / (i + 1)
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] "
                      f"avg_loss={avg:.4f} "
                      f"cls={loss_dict['classification'].item():.4f} "
                      f"reg={loss_dict['bbox_regression'].item():.4f}")

        lr_scheduler.step()

        val_loss = evaluate_loss(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(f"== Epoch {epoch} done in {elapsed:.1f}s | "
              f"train_loss={running_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss:.4f} ==")

        checkpoint_path = os.path.join(args.output_dir, f"retinanet_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()