import torch
from torch.utils.data import DataLoader

from dataset import FAIR1MDataset, collate_fn
from model import get_model, get_device
from classes import IDX_TO_CLASS


def main():
    device = get_device()

    print("Device:", device)

    # Load FAIR1M dataset
    dataset = FAIR1MDataset("../data")

    print("Dataset size:", len(dataset))

    # Load one image
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(loader))

    print("Image shape:", images[0].shape)
    print("Boxes shape:", targets[0]["boxes"].shape)
    print("Labels:", targets[0]["labels"])

    # Load RetinaNet
    model = get_model()

    checkpoint = torch.load(
       "../models/retinanet_epoch4.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("Loaded checkpoint: epoch 4")

    # Move image to MPS
    images = [image.to(device) for image in images]

    # Run inference
    print("\nRunning inference...")

    with torch.no_grad():
        predictions = model(images)

    prediction = predictions[0]

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    print("\nRaw predictions:", len(scores))

    if len(scores) > 0:
        print("Maximum confidence:", scores.max().item())
    else:
        print("Maximum confidence: 0")

    print("\nPredictions:")
    print("=" * 70)

    for box, label, score in zip(boxes, labels, scores):

        label_id = int(label.item())

        class_name = IDX_TO_CLASS.get(
            label_id,
            f"Unknown({label_id})"
        )

        print(
            f"Class: {class_name} | "
            f"ID: {label_id} | "
            f"Confidence: {score.item():.4f} | "
            f"Box: {box.tolist()}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()