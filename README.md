# VisionOrbit: Multi-Sensor All-Weather Building Footprint Segmentation

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20CUDA-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-SpaceNet%206-purple.svg)](https://spacenet.ai/sn6-challenge/)

**VisionOrbit** is a deep learning system designed for **all-weather, automated building footprint segmentation** by fusing complementary Earth observation modalities:
* **Optical Pan-Sharpened RGB (3 bands)**: High spatial resolution and semantic context, but vulnerable to cloud cover, precipitation, and nighttime.
* **Synthetic Aperture Radar / SAR (4 polarimetric bands)**: Active microwave imaging that penetrates dense cloud cover and operates day or night, but suffers from radar speckle noise and terrain geometry distortions.

By stacking RGB + SAR into **7-channel input tensors**, VisionOrbit learns invariant structural representations that enable robust building extraction across varying atmospheric and lighting conditions.

---

## Architecture Overview

```
                      ┌──────────────────────────────┐
                      │    Raw SpaceNet 6 Imagery    │
                      │  PS-RGB (3) + SAR-Intensity(4)│
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │   Multimodal Preprocessing   │
                      │  • SAR Percentile Scaling    │
                      │  • Building Polygon Rasterize│
                      │  • 256x256 Sliding Window    │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    7-Channel Input Tensor    │
                      │       (B, 7, 256, 256)       │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │       Multi-Sensor 2D U-Net           │
                 │  • 4-Stage DoubleConv Encoder         │
                 │  • Bottleneck (1024 channels)         │
                 │  • Decoder with Skip Concatenation    │
                 │  • Final 1x1 Conv Logit Output        │
                 └───────────────────┬───────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │   Training & Optimization │   │   Inference & Multi-Export│
       │   • BCE + Soft Dice Loss  │   │   • Binary PNG Masks      │
       │   • Cosine Annealing LR   │   │   • GeoJSON Vector Polys  │
       │   • Spatial Augmentations │   │   • Probability Heatmaps  │
       │   • Dataset-Level IoU/Dice│   │   • 4-Panel Visual Reports│
       └───────────────────────────┘   └───────────────────────────┘
```

---

## Project Structure

```
Ignite26/
├── data/
│   └── SpaceNet6/
│       ├── raw/                       # Extracted SpaceNet 6 Rotterdam (PS-RGB, SAR, GeoJSON)
│       ├── preprocess.py              # Ingestion, normalization, rasterization, and patch tiling
│       ├── visualize.py               # Preprocessing quality check & sample visualization
│       └── processed/                 # 90 preprocessed patches (256x256)
│           ├── images/                # 7-channel .npy arrays (RGB + SAR)
│           └── masks/                 # Binary .npy masks (0 or 1)
│
└── VisionOrbit/
    ├── dataset.py                     # PyTorch Dataset & DataLoader with spatial tile splitting
    ├── model.py                       # 7-channel input U-Net architecture (~31M parameters)
    ├── test_model.py                  # End-to-end integration and smoke test
    ├── train.py                       # Full training loop, hybrid loss, and checkpointing
    ├── predict.py                     # Multi-format prediction and diagnostic export pipeline
    ├── checkpoints/
    │   ├── best_model.pth             # Optimal model weights (Epoch 22, Val IoU: 0.5111)
    │   ├── latest_model.pth           # Final epoch snapshot
    │   └── training_history.json      # 25-epoch metrics trajectory
    ├── predictions/
    │   ├── masks/                     # Grayscale binary masks (PNG, 0/255)
    │   ├── vectors/                   # Individual patch GeoJSON building footprints
    │   ├── heatmaps/                  # Raw probability maps (.npy)
    │   ├── visualizations/            # 4-panel diagnostic comparison plots
    │   ├── all_predicted_buildings.geojson
    │   └── summary_report.json        # Quantitative test metrics & building counts
    └── README.md
```

---

## Data Preparation & Leakage Prevention

### 1. Robust SAR Normalization
SAR backscatter values exhibit a wide dynamic range and speckle noise. The preprocessing pipeline calculates the **2nd and 98th percentiles** per polarimetric band to scale radar intensity into $[0.0, 1.0]$ without outlier saturation:
$$\tilde{x} = \text{clip}\left(\frac{x - P_2}{P_{98} - P_2}, 0, 1\right)$$

### 2. Spatial Tile-Based Splitting
To prevent data leakage caused by spatial autocorrelation and overlapping patches, data is split **strictly by geographical tile ID**:
* **Training Set** (7 tiles: `55, 69, 783, 8137, 4164, 108, 442`): **63 patches**
* **Validation Set** (2 tiles: `7924, 2317`): **18 patches**
* **Test Set** (1 tile: `7218`): **9 patches**

---

## Loss Function & Evaluation Metrics

Satellite building footprints represent a minority class against large background areas. Relying solely on Binary Cross-Entropy (BCE) causes models to predict all-background pixels. VisionOrbit employs a **Hybrid Objective**:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p_i y_i) + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$

### Exact Dataset-Level Tracking
Instead of averaging per-batch scores (which skews scores on sparse tiles), the `MetricTracker` accumulates total True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$) across the entire dataset:
$$\text{IoU} = \frac{TP + \epsilon}{TP + FP + FN + \epsilon}, \quad \text{Dice} = \frac{2 \cdot TP + \epsilon}{2 \cdot TP + FP + FN + \epsilon}$$

---

## Performance Results

Trained on Apple Silicon (`MPS`) for 25 epochs:

| Metric | Initial (Epoch 1) | Best (Epoch 22) |
| :--- | :--- | :--- |
| **Train Loss** | `0.6976` | `0.4332` |
| **Train IoU** | `20.67%` | `50.97%` |
| **Validation Loss** | `0.6896` | **`0.4438`** |
| **Validation IoU** | `0.00%` | **`51.11%`** |
| **Validation Dice / F1** | `0.01%` | **`67.65%`** |
| **Average Epoch Time** | ~12.5 seconds | ~13.5 seconds |

---

## Quickstart & Usage

### 1. Environment Setup
Activate the project's virtual environment:
```bash
cd /Users/mehekdurgam/Ignite26/VisionOrbit
source ../.venv/bin/activate
```

### 2. Integration Smoke Test
Verify dataset loading, model forward pass, device compatibility, and loss backpropagation:
```bash
python test_model.py
```

### 3. Training the Model
Run the full training pipeline with spatial augmentations and Cosine Annealing:
```bash
python train.py --epochs 25 --batch-size 4 --lr 1e-4 --save-dir checkpoints
```

### 4. Running Inference & Generating Deliverables
Run predictions on unseen test tiles and export all output formats:
```bash
python predict.py --checkpoint checkpoints/best_model.pth --threshold 0.5
```

---

## Export Formats

`predict.py` generates four standard deliverables simultaneously:

| Format | Directory | Description | Use Case |
| :--- | :--- | :--- | :--- |
| **Binary Mask PNGs** | `predictions/masks/` | Grayscale $256 \times 256$ images ($0$ background, $255$ building). | Computer vision pipelines, OpenCV, web viewers. |
| **GeoJSON Vectors** | `predictions/vectors/` | GeoJSON `FeatureCollection` with individual building polygons, pixel area, and confidence. | QGIS, ArcGIS, Mapbox, official SpaceNet benchmark evaluation. |
| **Probability Heatmaps** | `predictions/heatmaps/` | Raw float32 NumPy arrays (`.npy`) containing continuous $[0.0, 1.0]$ probabilities. | Model ensembling, post-processing, calibration. |
| **Diagnostic Comparisons** | `predictions/visualizations/` | 4-panel visual reports comparing RGB, SAR, Ground Truth, and Predicted overlays. | Presentations, team review, qualitative inspection. |

---

## Future Roadmap

- [ ] **Multi-Sensor Ablation Study**: Train and compare Optical-Only (3 channels), SAR-Only (4 channels), and Fused (7 channels) to quantify multimodal fusion gains.
- [ ] **Simulated Cloud Masking**: Evaluate model degradation under synthetic optical occlusion to validate radar resilience.
- [ ] **Tile Mosaicking**: Stitch $256 \times 256$ patch predictions back into full-resolution geographic scenes.
