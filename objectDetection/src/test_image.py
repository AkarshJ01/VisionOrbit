import torch
import rasterio
from PIL import Image, ImageDraw

from model import get_model, get_device
from classes import IDX_TO_CLASS


# =========================
# SETTINGS
# =========================

IMAGE_PATH = "../data/images/162.tif"
CHECKPOINT = "../models/retinanet_epoch1.pth"
OUTPUT_PATH = "../outputs/test_162_epoch1.jpg"

CONFIDENCE_THRESHOLD = 0.30


def main():

    # -------------------------
    # Device
    # -------------------------

    device = get_device()
    print("Device:", device)

    # -------------------------
    # Load image
    # -------------------------

    print("\nLoading image...")

    with rasterio.open(IMAGE_PATH) as src:
        image_data = src.read()

        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("Data type:", src.dtypes)

    # FAIR1M images are RGB
    image_data = image_data[:3]

    # Convert CHW -> HWC
    image_data = image_data.transpose(1, 2, 0)

    # Convert to uint8 if necessary
    if image_data.dtype != "uint8":
        image_data = image_data.astype("uint8")

    image = Image.fromarray(image_data)

    # -------------------------
    # Prepare tensor
    # -------------------------

    image_tensor = torch.from_numpy(
        image_data.transpose(2, 0, 1)
    ).float() / 255.0

    image_tensor = image_tensor.to(device)

    # -------------------------
    # Load trained model
    # -------------------------

    print("\nLoading model...")

    model = get_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("Loaded:", CHECKPOINT)

    # -------------------------
    # Run detection
    # -------------------------

    print("\nRunning detection...")

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()
    
    print("Raw predictions:", len(scores))
    print("Maximum confidence:", scores.max().item() if len(scores) > 0 else 0)
    # -------------------------
    # Draw detections
    # -------------------------

    draw = ImageDraw.Draw(image)

    detection_count = 0

    print("\nDETECTIONS")
    print("=" * 70)

    for box, label, score in zip(boxes, labels, scores):

        score = float(score)

        if score < CONFIDENCE_THRESHOLD:
            continue

        detection_count += 1

        x1, y1, x2, y2 = box.tolist()

        label_id = int(label)

        class_name = IDX_TO_CLASS.get(
            label_id,
            f"Unknown({label_id})"
        )

        print(
            f"{detection_count}. "
            f"{class_name} | "
            f"confidence={score:.3f} | "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"{class_name} {score:.2f}",
            fill="red"
        )

    print("=" * 70)
    print("Total detections:", detection_count)

    # -------------------------
    # Save image
    # -------------------------

    import os

    os.makedirs(
        "../outputs",
        exist_ok=True
    )

    image.save(OUTPUT_PATH)

    print("\nResult saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
