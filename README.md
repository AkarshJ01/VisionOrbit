# VisionOrbit — SpaceNet 7 Urban Change Detection

## Overview

This repository contains the CNN-based multi-temporal change detection component of VisionOrbit.

The model takes two satellite images of the same geographic area at different points in time and detects areas that have undergone visual change, with a focus on likely new buildings and urban development.

The detected changes are converted into structured geographic events that can be passed to the RAG/LLM component of the VisionOrbit system.

## Pipeline

Before Image + After Image
        ↓
4-band satellite images
        ↓
8-channel input
        ↓
U-Net CNN
        ↓
Pixel-level change probability map
        ↓
Thresholding
        ↓
Connected-component detection
        ↓
Spatial merging and filtering
        ↓
Georeferencing
        ↓
Structured change events
        ↓
RAG / LLM

## Dataset

The model was trained using the SpaceNet 7 Multi-Temporal Urban Development dataset.

SpaceNet 7 contains multi-temporal satellite imagery and building annotations across multiple geographic areas.

Dataset: https://spacenet.ai/sn7-challenge/

## Model

The change detection model is a U-Net architecture implemented in PyTorch.

### Input

Two 4-band satellite images are combined into an 8-channel input:

- 4 bands from the earlier image
- 4 bands from the later image

### Output

The model produces a pixel-level change probability map.

Each pixel receives a value between 0 and 1 representing the model's estimated likelihood of visual change.

## Training

The model was trained using:

- Loss: BCEWithLogitsLoss + Dice Loss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 4
- Epochs: 25
- Framework: PyTorch

## Validation Performance

The best model achieved approximately:

- IoU: 19.51%
- Precision: 26.83%
- Recall: 41.68%
- Dice: 32.65%

on the held-out validation set.

A simple image-difference baseline achieved approximately 14.32% IoU, so the CNN performed better than the baseline on the validation data.

## Inference

The reusable inference pipeline is provided in:

`change_detector.py`

Example:

```python
from change_detector import detect_change

result = detect_change(
    "path/to/before.tif",
    "path/to/after.tif"
)

events = result["events"]

