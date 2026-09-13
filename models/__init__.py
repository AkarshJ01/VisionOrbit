"""Models module for VisionOrbit change detection architectures."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import DoubleConv, UNet

__all__ = ["DoubleConv", "UNet"]
