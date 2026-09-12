import torch
from torch.utils.data import DataLoader

from dataset import FAIR1MDataset, collate_fn
from model import get_model, get_device


def main():

    device = get_device()

    print("Device:", device)

    dataset = FAIR1MDataset("../data")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(loader))

    images = [image.to(device) for image in images]
    targets = [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]

    model = get_model()
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )

    print("\nRunning one training batch...")

    optimizer.zero_grad()

    loss_dict = model(images, targets)
    losses = sum(loss_dict.values())

    print("Classification loss:", loss_dict["classification"].item())
    print("Regression loss:", loss_dict["bbox_regression"].item())
    print("Total loss:", losses.item())

    losses.backward()

    grad = model.head.classification_head.cls_logits.weight.grad

    print("\nClassification head gradient:")

    if grad is None:
        print("ERROR: No gradient!")
    else:
        print("Gradient exists: YES")
        print("Gradient mean:", grad.mean().item())
        print("Gradient std:", grad.std().item())
        print("Gradient max:", grad.abs().max().item())

    optimizer.step()

    print("\nOptimizer step completed.")


if __name__ == "__main__":
    main()