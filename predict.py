import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio.features
from shapely.geometry import shape, mapping
import torch

from dataset import SpaceNetDataset, IMAGE_DIR, MASK_DIR
from model import UNet


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def calculate_patch_metrics(pred_mask, gt_mask, smooth=1e-6):
    """
    Computes IoU and Dice between predicted binary mask and ground truth.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total_pred = pred.sum()
    total_gt = gt.sum()

    union = total_pred + total_gt - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_pred + total_gt + smooth)

    return float(iou), float(dice)


def mask_to_geojson(binary_mask, prob_map, min_area=10):
    """
    Converts a binary raster mask into GeoJSON polygon features.
    Computes polygon area and mean confidence score for each building.
    """
    mask_uint8 = binary_mask.astype(np.uint8)
    features = []
    building_id = 1

    # Extract polygon geometries using rasterio shapes
    for geom_dict, value in rasterio.features.shapes(mask_uint8, mask=(mask_uint8 == 1)):
        poly = shape(geom_dict)

        if poly.area >= min_area:
            # Mask out the region inside this polygon to calculate average confidence
            geom_mask = rasterio.features.geometry_mask(
                [poly],
                out_shape=binary_mask.shape,
                transform=rasterio.transform.IDENTITY,
                invert=True
            )
            confidence = float(prob_map[geom_mask].mean()) if geom_mask.any() else 1.0

            feature = {
                "type": "Feature",
                "id": building_id,
                "properties": {
                    "building_id": building_id,
                    "area_pixels": round(float(poly.area), 2),
                    "confidence": round(confidence, 4)
                },
                "geometry": mapping(poly)
            }
            features.append(feature)
            building_id += 1

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return geojson, len(features)


def save_visual_comparison(rgb, sar, gt_mask, pred_mask, prob_map, iou, dice, save_path):
    """
    Saves a high-quality 4-panel diagnostic comparison plot:
    1. Optical RGB (True Color)
    2. SAR Radar (False color composite)
    3. Ground Truth Building Mask
    4. Model Prediction with Heatmap overlay & metrics
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: Optical RGB
    rgb_img = np.clip(np.transpose(rgb, (1, 2, 0)), 0, 1)
    axes[0].imshow(rgb_img)
    axes[0].set_title("Optical RGB (Pan-Sharpened)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: SAR Composite (using first 3 SAR bands as false-color RGB)
    sar_composite = np.clip(np.transpose(sar[:3], (1, 2, 0)), 0, 1)
    axes[1].imshow(sar_composite)
    axes[1].set_title("SAR Radar Intensity (All-Weather)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Ground Truth Mask
    axes[2].imshow(rgb_img)
    axes[2].imshow(gt_mask, cmap="Blues", alpha=0.55)
    axes[2].set_title(f"Ground Truth Footprints\n(Pixels: {int(gt_mask.sum()):,})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Model Prediction
    axes[3].imshow(rgb_img)
    axes[3].imshow(pred_mask, cmap="Oranges", alpha=0.55)
    title_text = f"Predicted Footprints\nIoU: {iou:.3f} | Dice: {dice:.3f}"
    axes[3].set_title(title_text, fontsize=11, fontweight="bold", color="darkred" if iou < 0.3 else "darkgreen")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# Main Inference Pipeline
# --------------------------------------------------

def run_predictions(
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="predictions",
    threshold=0.5,
    min_area=10,
    test_tiles=None
):
    if test_tiles is None:
        test_tiles = [7218]  # Default unseen test tile

    # Hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 65)
    print("VISIONORBIT: MULTI-SENSOR PREDICTION & EXPORT PIPELINE")
    print("=" * 65)
    print(f"Device:           {device}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Decision Threshold: {threshold}")
    print(f"Min Building Area:  {min_area} pixels")
    print(f"Test Tiles:       {test_tiles}")
    print("=" * 65)

    # Setup output directories
    out_dir = Path(output_dir)
    masks_dir = out_dir / "masks"
    vectors_dir = out_dir / "vectors"
    heatmaps_dir = out_dir / "heatmaps"
    viz_dir = out_dir / "visualizations"

    for d in [masks_dir, vectors_dir, heatmaps_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load Model
    model = UNet(in_channels=7, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint.get("epoch", "N/A")
        ckpt_iou = checkpoint.get("val_iou", "N/A")
        print(f"Loaded weights from Epoch {ckpt_epoch} (Val IoU: {ckpt_iou:.4f})")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load Test Dataset
    dataset = SpaceNetDataset(IMAGE_DIR, MASK_DIR, test_tiles)
    print(f"Found {len(dataset)} test patches to predict.\n")

    summary_records = []
    total_buildings = 0
    all_features = []

    print(f"{'Patch':<22} | {'Buildings':<10} | {'IoU':<8} | {'Dice':<8} | {'Status'}")
    print("-" * 65)

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, mask_tensor = dataset[idx]
            image_path, mask_path = dataset.samples[idx]
            stem = image_path.stem

            # Forward pass
            inputs = image_tensor.unsqueeze(0).to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Binary threshold
            pred_mask = (probs > threshold).astype(np.uint8)
            gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

            # Compute metrics
            iou, dice = calculate_patch_metrics(pred_mask, gt_mask)

            # 1. Save Binary Mask Image (PNG, 0 or 255)
            mask_png_path = masks_dir / f"{stem}_mask.png"
            Image.fromarray(pred_mask * 255).save(mask_png_path)

            # 2. Save Probability Heatmap (.npy)
            heatmap_path = heatmaps_dir / f"{stem}_prob.npy"
            np.save(heatmap_path, probs.astype(np.float32))

            # 3. Export Vector Polygons (GeoJSON)
            geojson_data, num_buildings = mask_to_geojson(pred_mask, probs, min_area=min_area)
            geojson_path = vectors_dir / f"{stem}_buildings.geojson"
            with open(geojson_path, "w") as f:
                json.dump(geojson_data, f, indent=2)

            for feat in geojson_data["features"]:
                feat_copy = dict(feat)
                feat_copy["properties"]["patch"] = stem
                all_features.append(feat_copy)

            total_buildings += num_buildings

            # 4. Save 4-Panel Visualization Plot (PNG)
            image_np = image_tensor.numpy()
            rgb = image_np[:3]
            sar = image_np[3:7]
            viz_path = viz_dir / f"{stem}_eval.png"
            save_visual_comparison(rgb, sar, gt_mask, pred_mask, probs, iou, dice, viz_path)

            summary_records.append({
                "patch": stem,
                "buildings_detected": num_buildings,
                "iou": round(iou, 4),
                "dice": round(dice, 4),
                "ground_truth_pixels": int(gt_mask.sum()),
                "predicted_pixels": int(pred_mask.sum())
            })

            print(f"{stem:<22} | {num_buildings:<10d} | {iou:<8.4f} | {dice:<8.4f} | Saved")

    # Save combined GeoJSON with all detected buildings
    combined_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }
    with open(out_dir / "all_predicted_buildings.geojson", "w") as f:
        json.dump(combined_geojson, f, indent=2)

    # Save overall summary JSON
    mean_iou = float(np.mean([r["iou"] for r in summary_records]))
    mean_dice = float(np.mean([r["dice"] for r in summary_records]))

    final_summary = {
        "total_test_patches": len(dataset),
        "mean_test_iou": round(mean_iou, 4),
        "mean_test_dice": round(mean_dice, 4),
        "total_buildings_detected": total_buildings,
        "patches": summary_records
    }

    with open(out_dir / "summary_report.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("-" * 65)
    print(f"\nInference Complete!")
    print(f"Mean Test IoU:            {mean_iou:.4f}")
    print(f"Mean Test Dice:           {mean_dice:.4f}")
    print(f"Total Buildings Detected: {total_buildings}")
    print(f"\nOutputs exported to: {out_dir.resolve()}")
    print(f"  ├── masks/          (Binary 0/255 PNG masks)")
    print(f"  ├── vectors/        (GeoJSON building footprints)")
    print(f"  ├── heatmaps/       (Raw probability float32 .npy maps)")
    print(f"  ├── visualizations/ (4-panel diagnostic comparison PNGs)")
    print(f"  ├── all_predicted_buildings.geojson")
    print(f"  └── summary_report.json")


# --------------------------------------------------
# CLI Entry Point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run VisionOrbit Multi-Sensor Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Output directory for generated predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for building mask (default: 0.5)")
    parser.add_argument("--min-area", type=int, default=10, help="Minimum pixel area for polygon filtering (default: 10)")
    parser.add_argument("--tiles", nargs="+", type=int, default=[7218], help="List of tile IDs to evaluate")
    args = parser.parse_args()

    run_predictions(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_area=args.min_area,
        test_tiles=args.tiles
    )


if __name__ == "__main__":
    main()
