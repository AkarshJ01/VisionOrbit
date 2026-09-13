"""Inference module for VisionOrbit change detection."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.change_detector import (
    DEVICE,
    CHECKPOINT,
    THRESHOLD,
    PATCH_SIZE,
    detect_change,
    load_model,
    load_image,
    predict_probability_map,
    extract_regions,
    create_events,
    merge_regions,
)

__all__ = [
    "DEVICE",
    "CHECKPOINT",
    "THRESHOLD",
    "PATCH_SIZE",
    "detect_change",
    "load_model",
    "load_image",
    "predict_probability_map",
    "extract_regions",
    "create_events",
    "merge_regions",
]
