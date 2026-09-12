import torch
from torch.utils.data import DataLoader

from dataset import FAIR1MDataset, collate_fn
from model import get_model, get_device


def main():

    device = get_device()

    print("Device:", device)

    # Dataset
    dataset = FAIR1MDataset("../data")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(loader))

    image = images[0].to(device)

    print("Image shape:", image.shape)
    print("Ground truth boxes:", targets[0]["boxes"].shape)
    print("Ground truth labels:", targets[0]["labels"])

    # Load trained model
    model = get_model()

    checkpoint = torch.load(
        "../models/retinanet_epoch4.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("Loaded epoch:", checkpoint["epoch"])

    # Get backbone features
    with torch.no_grad():

        features = model.backbone(image.unsqueeze(0))

        if isinstance(features, torch.Tensor):
            features = {"0": features}

        print("\nFeature maps:")

        for name, feature in features.items():
            print(
                name,
                feature.shape,
                "min=", feature.min().item(),
                "max=", feature.max().item()
            )

        # RetinaNet head
        head_outputs = model.head(
            list(features.values())
        )

        print("\nHead output keys:")
        print(head_outputs.keys())

        print("\nClassification output shape:")
        print(head_outputs["cls_logits"].shape)

        cls_logits = head_outputs["cls_logits"]
        

        print(
            "Logit min:",
            cls_logits.min().item()
        )

        print(
            "Logit max:",
            cls_logits.max().item()
        )

        print(
            "Logit mean:",
            cls_logits.mean().item()
        )


if __name__ == "__main__":
    main()