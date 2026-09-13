<<<<<<< HEAD
# VisionOrbit: Multi-Sensor All-Weather Building Footprint Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20CUDA%20%7C%20CPU-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

**VisionOrbit** is a deep learning system for **all-weather, automated building footprint extraction** from multimodal satellite imagery. It fuses two complementary Earth observation sensors:

* **Optical Pan-Sharpened RGB (3 bands)**: High spatial resolution and color contrast, but obscured by clouds, heavy precipitation, and nighttime.
* **Synthetic Aperture Radar / SAR (4 polarimetric bands)**: Active microwave imaging that penetrates dense cloud cover and functions day or night, but contains speckle noise.

By stacking RGB + SAR into a **7-channel input tensor**, VisionOrbit learns invariant structural features to accurately segment building footprints even when optical visibility is degraded.

Pretrained model weights and sample data are included directly in this repository so you can run predictions immediately.

---

## Quickstart (Run in 30 Seconds)

### 1. Clone the Repository
```bash
git clone https://github.com/AkarshJ01/VisionOrbit.git
cd VisionOrbit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference with Pretrained Weights
Run predictions immediately using the included pretrained weights and demo patch:
```bash
python predict.py
```
Outputs are automatically generated and saved to the `predictions/` directory:
* `predictions/visualizations/` &mdash; 4-panel diagnostic comparison plots (RGB, SAR, Ground Truth, Predicted Overlay)
* `predictions/masks/` &mdash; Grayscale binary mask PNGs (0 = background, 255 = building)
* `predictions/vectors/` &mdash; GeoJSON polygon files for GIS viewers (QGIS, ArcGIS, Mapbox)
* `predictions/heatmaps/` &mdash; Raw continuous probability `.npy` arrays
* `predictions/all_predicted_buildings.geojson` &mdash; Consolidated vector polygons for all detected buildings
* `predictions/summary_report.json` &mdash; Evaluation metrics & detected building counts

---

## How It Works

```
              ┌──────────────────────────────────────┐
              │    Multimodal Satellite Input        │
              │  3 Optical RGB  +  4 SAR Radar Bands │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │        7-Channel Input Tensor        │
              │           (B, 7, 256, 256)           │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
             ┌──────────────────────────────────────────┐
             │       Multi-Sensor 2D U-Net              │
             │  • 4-Stage DoubleConv Encoder (64→512)   │
             │  • Bottleneck Feature Map (1024 channels)│
             │  • Decoder with Skip Concatenation       │
             │  • Final 1x1 Conv Logit Output           │
             └───────────────────┬──────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────┐
   │    Raster Deliverables    │   │     GIS & Analytics       │
   │  • Binary Mask (0 or 255) │   │  • Vector GeoJSON Polygons│
   │  • Probability Heatmaps   │   │  • Surface Area & Counts  │
   │  • 4-Panel Report PNGs    │   │  • Building Confidence    │
   └───────────────────────────┘   └───────────────────────────┘
```

---

## Repository Structure

```
VisionOrbit/
├── demo_data/                     # Ready-to-use multimodal sample data
│   ├── sample_image.npy           # 7-channel test patch (3 RGB + 4 SAR)
│   └── sample_mask.npy            # Ground-truth binary building mask
├── weights/                       # Pretrained model weights
│   └── best_model.pth             # Pretrained U-Net weights (IoU: 51.11%, Dice: 67.65%)
├── scripts/                       # Data processing & visualization utilities
│   ├── preprocess.py              # Multimodal normalization, rasterization, and tiling
│   └── visualize.py               # Visual inspection script
├── model.py                       # 7-channel input U-Net architecture (~31M parameters)
├── dataset.py                     # PyTorch Dataset loader with dynamic data discovery
├── test_model.py                  # Integration smoke test for hardware & forward/backward pass
├── train.py                       # Training pipeline with BCE+Dice loss and augmentations
├── predict.py                     # Multi-format prediction and diagnostic export pipeline
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies list
└── README.md                      # Project documentation
```

---

## Advanced Usage

### Running Predictions on Custom Images or Folders
Predict on any 7-channel `.npy` patch or directory of patches:
```bash
# Predict on a single image file
python predict.py --input /path/to/my_patch.npy

# Predict on an entire folder of patches
python predict.py --input /path/to/my_patches_folder/

# Adjust decision threshold (default: 0.5) and minimum building area
python predict.py --threshold 0.45 --min-area 15
```

### Verification Smoke Test
Run an end-to-end integration check to test your GPU/CPU hardware and model pipeline:
```bash
python test_model.py
```

### Training on Custom Datasets
Train the U-Net from scratch or resume training:
```bash
# Basic training
python train.py --epochs 25 --batch-size 4 --lr 1e-4

# Specify custom data directory and checkpoint output
python train.py --data-dir /path/to/data --save-dir checkpoints --epochs 30
```

---

## Model Performance

Trained for 25 epochs using a hybrid **BCEWithLogitsLoss + Soft Dice Loss** with spatial data augmentations and Cosine Annealing learning rate schedule:

| Metric | Baseline (Epoch 1) | Best Model (Epoch 22) |
| :--- | :--- | :--- |
| **Validation IoU (Jaccard Index)** | `0.00%` | **`51.11%`** |
| **Validation Dice / F1-Score** | `0.01%` | **`67.65%`** |
| **Validation Loss** | `0.6896` | **`0.4438`** |
| **Training Speed** | ~12 seconds / epoch (Apple Silicon MPS) | |

---

## Supported Deliverables

When running `predict.py`, the following four standard formats are exported simultaneously:

| Format | Output Path | Description | Best For |
| :--- | :--- | :--- | :--- |
| **Binary Mask Images** | `predictions/masks/` | Grayscale $256 \times 256$ PNGs (`0` background, `255` building). | Computer vision pipelines, OpenCV, frontend web apps. |
| **Vector Polygons** | `predictions/vectors/` | GeoJSON `FeatureCollection` containing building polygons, areas, and confidence. | QGIS, ArcGIS, Mapbox, urban planning spatial queries. |
| **Probability Heatmaps** | `predictions/heatmaps/` | Raw float32 NumPy arrays (`.npy`) containing continuous $[0.0, 1.0]$ confidence values. | Model calibration, ensemble modeling, custom thresholds. |
| **Diagnostic Reports** | `predictions/visualizations/` | High-resolution 4-panel visual plots comparing RGB, SAR, Ground Truth, and Predictions. | Slide decks, executive reports, model inspection. |

---

## License
This project is open-source under the MIT License.
=======
# 🪐 VisionOrbit

**VisionOrbit** is a modern, multimodal AI visual intelligence web application inspired by OpenAI's clean conversational interface. It enables users to ask questions, prompt the AI, upload images (via drag-and-drop, file picker, or clipboard paste `⌘V`), and receive in-depth visual analyses, OCR text extraction, UI/UX breakdowns, and telemetry insights.

---

## 🌟 Key Features

1. **OpenAI-Style Prompting Handle & Interface**:
   - Clean obsidian dark theme with smooth glassmorphism and subtle neon glow accents.
   - Auto-expanding multiline prompt input dock with keyboard shortcuts (`Enter` to submit, `Shift+Enter` for new lines, `⌘K` for New Chat).
   - Real-time animated streaming response cards with markdown formatting, syntax-highlighted code blocks, telemetry tables, and one-click copy/speak actions.
   - Welcoming Hero interface with quick starter suggestion cards for deep visual inspection, OCR transcription, UI design review, and chart/graph analytics.

2. **Multimodal Image Analysis & Staging**:
   - **Multiple Upload Channels**:
     - Drag & drop images anywhere on the window or into the hero dropzone.
     - Direct clipboard paste (`Cmd+V` / `Ctrl+V`) for instant screenshot analysis.
     - Attachment button (`📎`) supporting `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`.
   - Staging tray showing image thumbnail cards before sending.
   - Click-to-zoom interactive **Lightbox Modal** for full-resolution inspection.

3. **Multi-Provider Neural Architecture**:
   - **OpenAI Multimodal Vision** (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`): High-intelligence live vision reasoning.
   - **Ollama Local Vision** (`llava`, `llama3.2-vision`, `moondream`, etc.): On-device private multimodal intelligence.
   - **VisionOrbit Intelligence Engine**: Fast built-in vision telemetry and heuristic analysis engine that works out-of-the-box with zero API key dependencies.
   - **Tavily Web Search Enrichment**: Real-time web fact-checking for multimodal queries.

4. **Chat History & Management**:
   - Persistent conversation threads saved in browser `localStorage`.
   - Export chats as Markdown (`.md`).
   - Theme switch between Obsidian Dark and Clean Slate Light.

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.13+ and `uv` are installed:

```bash
# Install dependencies
uv sync
```

### 2. Start the VisionOrbit Application

Run the server:

```bash
uv run python main.py
```

The application will start immediately at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ⚙️ Configuration (Optional)

You can configure API keys either via a `.env` file or directly inside the web UI via **Settings (⚙️)**:

```ini
# .env (Optional)
PORT=8000
OPENAI_API_KEY=sk-proj-...
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=tvly-...
```

---

## 🧪 Testing Backend API

To verify backend multimodal processing and telemetry extraction:

```bash
uv run python test_app.py
```
>>>>>>> origin/main
