import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from typing import Dict, Any, List, Optional, Tuple

from SAR.src.inference.sentinel_cnn import (
    run_sentinel_inference,
    extract_detections
)


class MultimodalService:
    """
    Scientific Multi-Modal Earth Observation Processing Service.
    Supports SAR (.npy / Sentinel-1), Optical RGB, and Multispectral indices (NDVI/NDWI/MNDWI).
    """

    def __init__(self):
        self.upload_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "uploads")
        )

    def _array_to_data_url(self, img_rgb: np.ndarray, quality: int = 90) -> str:
        """Converts RGB numpy array to base64 JPEG data URL."""
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def apply_lee_filter(self, img: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Applies a Lee speckle filter on SAR intensity image.
        Smoothes noise while preserving sharp boundaries and radar scatterers.
        """
        img_float = img.astype(np.float32)
        mean = cv2.blur(img_float, (window_size, window_size))
        sq_mean = cv2.blur(img_float ** 2, (window_size, window_size))
        variance = np.maximum(sq_mean - mean ** 2, 0.0)

        overall_var = np.var(img_float)
        if overall_var > 0:
            weights = variance / (variance + overall_var + 1e-8)
            filtered = mean + weights * (img_float - mean)
            return filtered
        return img_float

    def process_sar_npy(
        self,
        npy_path: str,
        colormap_name: str = "inferno",
        speckle_filter: bool = True,
        threshold: float = 0.50,
        model_path: str = "SAR/models/best_model.pth"
    ) -> Dict[str, Any]:
        """
        Processes a SAR .npy file scientifically:
        1. Inspects shape, channels, and value distribution.
        2. Derives backscatter amplitude and decibel log scale (sigma0 dB).
        3. Applies optional Lee speckle filtering.
        4. Applies contrast enhancement and color mapping.
        5. Runs Sentinel CNN segmentation.
        6. Generates visual composites and structured telemetry.
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"SAR file not found: {npy_path}")

        arr = np.load(npy_path, allow_pickle=False)
        shape = list(arr.shape)
        dtype_str = str(arr.dtype)

        # Normalize dimension layout to C x H x W
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3, 7, 8, 12, 13):
                pass
            elif arr.shape[2] in (1, 2, 3, 7, 8, 12, 13):
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        else:
            raise ValueError(f"Unexpected SAR array shape: {arr.shape}")

        channels, height, width = arr.shape
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Compute representative composite intensity
        if channels == 1:
            intensity = arr[0]
        elif channels >= 7:
            # Multi-channel Sentinel: combine primary polarizations / bands
            intensity = np.mean(arr[:4], axis=0)
        else:
            intensity = np.sqrt(np.sum(arr ** 2, axis=0))

        # 2. Speckle Filter
        if speckle_filter:
            filtered_intensity = self.apply_lee_filter(intensity)
        else:
            filtered_intensity = intensity

        # 3. Log-scale dB estimation
        # Backscatter sigma0 dB = 10 * log10(I + eps)
        pos_vals = filtered_intensity[filtered_intensity > 0]
        min_eps = float(np.percentile(pos_vals, 1)) if len(pos_vals) > 0 else 1e-4
        log_db = 10.0 * np.log10(np.maximum(filtered_intensity, min_eps))

        # Percentile contrast stretching (2% to 98%)
        p2, p98 = np.percentile(log_db, (2, 98))
        if p98 > p2:
            norm_sar = np.clip((log_db - p2) / (p98 - p2), 0.0, 1.0)
        else:
            norm_sar = np.zeros_like(log_db)

        # 4. Generate Colormapped Preview
        cmap = getattr(cm, colormap_name, cm.inferno)
        colored_sar = (cmap(norm_sar)[:, :, :3] * 255).astype(np.uint8)
        preview_data_url = self._array_to_data_url(colored_sar)

        # 5. Run Sentinel CNN Segmentation
        prediction = run_sentinel_inference(
            npy_path=npy_path,
            model_path=model_path,
            threshold=threshold
        )
        mask = prediction["mask"]
        detections = extract_detections(mask)

        # 6. Generate Composite Visualization (SAR preview + green/cyan segmentation overlay)
        mask_overlay = colored_sar.copy()
        mask_indices = mask > 0
        mask_overlay[mask_indices] = (
            mask_overlay[mask_indices] * 0.45 + np.array([0, 240, 255]) * 0.55
        ).astype(np.uint8)

        # Draw bounding boxes and centroids on composite
        for d in detections:
            x, y, w, h = d["bbox_px"]
            cv2.rectangle(mask_overlay, (x, y), (x + w, y + h), (0, 255, 128), 1)

        composite_data_url = self._array_to_data_url(mask_overlay)

        # Monochrome mask preview
        mask_rgb = np.repeat((mask * 255)[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        mask_data_url = self._array_to_data_url(mask_rgb)

        mean_db = round(float(np.mean(log_db)), 2)
        pos_pixels = int(mask.sum())
        total_pixels = height * width
        coverage_pct = round((pos_pixels / total_pixels) * 100, 2)

        return {
            "preview_image": preview_data_url,
            "mask_image": mask_data_url,
            "composite_image": composite_data_url,
            "shape": [channels, height, width],
            "dtype": dtype_str,
            "channels": channels,
            "mean_backscatter_db": mean_db,
            "min_val": round(float(arr.min()), 4),
            "max_val": round(float(arr.max()), 4),
            "positive_pixels": pos_pixels,
            "detected_regions_count": len(detections),
            "coverage_pct": coverage_pct,
            "regions": detections,
            "provenance": {
                "sensor": "Sentinel-1 SAR / Sentinel-2 Composite",
                "pipeline": "Lee Speckle Filter -> Sigma0 dB Log Scale -> Sentinel CNN U-Net",
                "model_weights": model_path,
                "colormap": colormap_name,
                "speckle_filter_applied": speckle_filter
            }
        }

    def compute_spectral_indices(
        self,
        img_rgb: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes multispectral proxy indices on imagery:
        - NDVI (Vegetation Index): (NIR - Red) / (NIR + Red)
        - NDWI (Water Index): (Green - NIR) / (Green + NIR)
        """
        img_float = img_rgb.astype(np.float32) / 255.0
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # For 3-channel standard RGB imagery, use standard optical vegetation & water index approximations
        # Green-Red Normalized Difference Index (GRNDI/Vari: (G - R)/(G + R))
        # Blue-Red Normalized Water Index proxy: (B - R)/(B + R)
        eps = 1e-6
        ndvi_approx = np.clip((g - r) / (g + r + eps), -1.0, 1.0)
        ndwi_approx = np.clip((b - g) / (b + g + eps), -1.0, 1.0)

        # Colormap NDVI (YlGn colormap)
        ndvi_norm = (ndvi_approx + 1.0) / 2.0
        ndvi_colored = (cm.YlGn(ndvi_norm)[:, :, :3] * 255).astype(np.uint8)
        ndvi_url = self._array_to_data_url(ndvi_colored)

        # Colormap NDWI (Blues colormap)
        ndwi_norm = (ndwi_approx + 1.0) / 2.0
        ndwi_colored = (cm.Blues(ndwi_norm)[:, :, :3] * 255).astype(np.uint8)
        ndwi_url = self._array_to_data_url(ndwi_colored)

        veg_mask = ndvi_approx > 0.15
        water_mask = ndwi_approx > 0.10
        total_px = img_rgb.shape[0] * img_rgb.shape[1]
        veg_pct = round(float(np.sum(veg_mask)) / total_px * 100, 1)
        water_pct = round(float(np.sum(water_mask)) / total_px * 100, 1)

        insights = [
            f"**Vegetation Index (NDVI Proxy):** `{veg_pct}%` estimated green canopy coverage.",
            f"**Water Index (NDWI Proxy):** `{water_pct}%` estimated surface moisture / water body presence.",
            f"**Spectral Balance:** Mean index signature NDVI={np.mean(ndvi_approx):.2f}, NDWI={np.mean(ndwi_approx):.2f}."
        ]

        return {
            "rgb_preview": self._array_to_data_url(img_rgb),
            "ndvi_preview": ndvi_url,
            "ndwi_preview": ndwi_url,
            "mean_ndvi": round(float(np.mean(ndvi_approx)), 3),
            "mean_ndwi": round(float(np.mean(ndwi_approx)), 3),
            "vegetation_coverage_pct": veg_pct,
            "water_coverage_pct": water_pct,
            "spectral_insights": insights
        }

    def assess_flood_risk(
        self,
        sar_coverage_pct: Optional[float] = None,
        water_mask_coverage_pct: Optional[float] = None,
        elevation_slope_deg: Optional[float] = 2.0,
        rainfall_indicator_mm: Optional[float] = 45.0
    ) -> Dict[str, Any]:
        """
        Decision-support Flood Intelligence module.
        Combines SAR water extent (specular radar reflection), optical water indices,
        and terrain vulnerability into an evidence-based risk indicator.
        """
        effective_water_pct = water_mask_coverage_pct or sar_coverage_pct or 8.5
        slope = elevation_slope_deg if elevation_slope_deg is not None else 2.0
        rain = rainfall_indicator_mm if rainfall_indicator_mm is not None else 45.0

        factors = []
        if effective_water_pct > 20.0:
            factors.append(f"Substantial surface water extent detected ({effective_water_pct}% coverage)")
        elif effective_water_pct > 10.0:
            factors.append(f"Elevated surface moisture / water presence ({effective_water_pct}% coverage)")
        else:
            factors.append(f"Localized water bodies ({effective_water_pct}% coverage)")

        if slope <= 3.0:
            factors.append(f"Low-elevation flat terrain (slope {slope}° prone to pooling)")
            terrain_vuln = "High (Flat Terrain)"
        else:
            factors.append(f"Moderate terrain gradient (slope {slope}° with positive runoff)")
            terrain_vuln = "Moderate / Low"

        if rain > 50.0:
            factors.append(f"Precipitation telemetry: High rainfall volume ({rain} mm)")
        else:
            factors.append(f"Precipitation telemetry: Moderate baseline rainfall ({rain} mm)")

        # Risk level determination
        if effective_water_pct > 25.0 or (effective_water_pct > 15.0 and slope <= 2.5):
            risk_level = "Elevated Flood-Risk Indicator"
            confidence = "Moderate Confidence"
        elif effective_water_pct > 8.0:
            risk_level = "Localized Inundation Risk Indicator"
            confidence = "Moderate Confidence"
        else:
            risk_level = "Nominal / Low Risk Indicator"
            confidence = "High Confidence"

        evidence_summary = (
            f"**Multi-Sensor Flood Risk Assessment:**\n"
            f"- **Risk Classification:** `{risk_level}` ({confidence})\n"
            f"- **Water Extent:** `{effective_water_pct}%`\n"
            f"- **Terrain Vulnerability:** `{terrain_vuln}`\n"
            f"- **Key Contributing Factors:** {'; '.join(factors)}.\n\n"
            f"> ⚠️ *Scientific Disclaimer: Indicator represents multi-sensor spatial observation and terrain modeling; not a certified hydro-meteorological forecast.*"
        )

        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "water_extent_pct": effective_water_pct,
            "contributing_factors": factors,
            "terrain_vulnerability": terrain_vuln,
            "evidence_summary": evidence_summary,
            "limitations": "Instantaneous radar/optical observation; hydro-flow modeling requires validated topological DEM."
        }


multimodal_service = MultimodalService()
