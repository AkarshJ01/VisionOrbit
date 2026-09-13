"""Backward compatibility module for model.py imports."""
from models.unet import DoubleConv, UNet

__all__ = ["DoubleConv", "UNet"]
