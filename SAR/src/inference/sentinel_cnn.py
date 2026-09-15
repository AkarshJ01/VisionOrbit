import torch
import torch.nn as nn
import numpy as np
import cv2


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SentinelCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 7-channel Sentinel input
        self.enc1 = DoubleConv(7, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(
            1024, 512
        )

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(
            512, 256
        )

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(
            256, 128
        )

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(
            128, 64
        )

        self.final = nn.Conv2d(
            64, 1, 1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        return self.final(d1)


def load_sentinel_model(
    model_path="SAR/models/best_model.pth"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SentinelCNN()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device


def prepare_npy(path):

    arr = np.load(
        path,
        allow_pickle=False
    )

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D NPY, got {arr.shape}"
        )

    # C x H x W
    if arr.shape[0] == 7:
        pass

    # H x W x C
    elif arr.shape[2] == 7:
        arr = np.transpose(
            arr,
            (2, 0, 1)
        )

    else:
        raise ValueError(
            f"Expected 7 channels, got {arr.shape}"
        )

    arr = arr.astype(
        np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize each channel
    for i in range(7):

        channel = arr[i]

        min_val = np.min(channel)
        max_val = np.max(channel)

        if max_val > min_val:

            arr[i] = (
                channel - min_val
            ) / (
                max_val - min_val
            )

        else:

            arr[i] = 0.0

    return arr


def run_sentinel_inference(
    npy_path,
    model_path="SAR/models/best_model.pth",
    threshold=0.5
):

    model, device = load_sentinel_model(
        model_path
    )

    image = prepare_npy(
        npy_path
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        )

    probability = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    return {
        "mask": mask,
        "probability": probability,
        "input_shape": list(tensor.shape),
        "device": str(device)
    }


def extract_detections(
    mask,
    min_area=20
):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    detections = []

    for i in range(1, num_labels):

        x = int(
            stats[i, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[i, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[i, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[i, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < min_area:
            continue

        cx = float(
            centroids[i][0]
        )

        cy = float(
            centroids[i][1]
        )

        detections.append({
            "id": len(detections) + 1,
            "class": "detected_region",
            "area_px": area,
            "center_px": [
                round(cx, 1),
                round(cy, 1)
            ],
            "bbox_px": [
                x,
                y,
                w,
                h
            ]
        })

    return detections