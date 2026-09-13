import os
import io
import sys
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure objectDetection directory is in Python path
OBJECT_DETECTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "objectDetection"))
if OBJECT_DETECTION_DIR not in sys.path:
    sys.path.insert(0, OBJECT_DETECTION_DIR)

from src.inference.tiled_inference import TiledOBBInferrer
from src.inference.georeference import (
    GeoReferenceEngine,
    calculate_geo_distance,
    format_distance
)
from src.inference.predict import compute_obb_attributes, load_image_rgb

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class DetectionService:
    def __init__(self, model_relative_path: str = "models/best.pt", tile_size: int = 640):
        self.model_path = os.path.join(OBJECT_DETECTION_DIR, model_relative_path)
        self.tile_size = tile_size
        self._inferrer: Optional[TiledOBBInferrer] = None

    def _get_inferrer(self) -> TiledOBBInferrer:
        """Lazy load the YOLO-OBB inference engine once."""
        if self._inferrer is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
            print(f"[DetectionService] Loading YOLO-OBB model weights from: {self.model_path}")
            self._inferrer = TiledOBBInferrer(model_path=self.model_path, tile_size=self.tile_size)
            print("[DetectionService] Model loaded successfully.")
        return self._inferrer

    def decode_image_data(self, image_input: Union[str, bytes]) -> Tuple[np.ndarray, Optional[str], int, int]:
        """
        Decodes image from base64 string, data URL, file path, or raw bytes into an RGB numpy uint8 array.
        Returns: (img_rgb, original_format, width, height)
        """
        # If given a file path on disk
        if isinstance(image_input, str) and not image_input.startswith("data:") and os.path.exists(os.path.expanduser(image_input)):
            expanded_path = os.path.expanduser(image_input)
            img_rgb = load_image_rgb(expanded_path)
            h, w = img_rgb.shape[:2]
            fmt = os.path.splitext(expanded_path)[1].lstrip(".").upper() or "TIFF"
            return img_rgb, fmt, w, h

        # If base64 data URL or raw base64 string
        raw_bytes: bytes
        fmt = "PNG"
        if isinstance(image_input, str):
            if image_input.startswith("data:"):
                header, encoded = image_input.split(",", 1)
                if "image/tiff" in header or "image/tif" in header:
                    fmt = "TIFF"
                elif "image/jpeg" in header or "image/jpg" in header:
                    fmt = "JPEG"
                elif "image/webp" in header:
                    fmt = "WEBP"
                raw_bytes = base64.b64decode(encoded)
            else:
                raw_bytes = base64.b64decode(image_input)
        else:
            raw_bytes = image_input

        # Try tifffile if TIFF or generic
        if HAS_TIFFFILE and (fmt == "TIFF" or raw_bytes.startswith(b'II*\x00') or raw_bytes.startswith(b'MM\x00*')):
            try:
                with io.BytesIO(raw_bytes) as bio:
                    img = tifffile.imread(bio)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif len(img.shape) == 3:
                        if img.shape[2] >= 3:
                            img = img[:, :, :3]
                        elif img.shape[2] == 1:
                            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
                    if img.dtype != np.uint8:
                        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    h, w = img.shape[:2]
                    return img, "TIFF", w, h
            except Exception as e:
                print(f"[DetectionService] tifffile decoding warning: {e}")

        # Fallback to OpenCV / PIL
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            return img_rgb, fmt, w, h

        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"))
            h, w = img_rgb.shape[:2]
            return img_rgb, pil_img.format or fmt, w, h

    def encode_image_to_data_url(self, img_rgb: np.ndarray, quality: int = 90) -> str:
        """Encodes an RGB numpy array to a web-compatible base64 JPEG data URL."""
        pil_img = Image.fromarray(img_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=quality)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def detect(
        self,
        image_input: Union[str, bytes],
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        use_tiling: bool = True,
        enhance_contrast: bool = True,
        multi_scale_fusion: bool = True,
        reference_lat: Optional[float] = None,
        reference_lon: Optional[float] = None,
        temp_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs high-accuracy YOLO-OBB dynamic object detection on the provided image input.
        Returns:
            - annotated_image: base64 data URL of image with bounding boxes
            - preview_image: base64 data URL of original image (RGB converted)
            - detections: list of enriched detected objects
            - summary: class breakdown, count, resolution
            - formatted_context: markdown string for LLM system prompt / visual context
        """
        inferrer = self._get_inferrer()
        img_rgb, fmt, w, h = self.decode_image_data(image_input)

        # Inspect georeferencing if file path is available
        geo_engine = None
        if isinstance(image_input, str) and os.path.exists(os.path.expanduser(image_input)):
            geo_engine = GeoReferenceEngine(os.path.expanduser(image_input))
        elif temp_file_path and os.path.exists(temp_file_path):
            geo_engine = GeoReferenceEngine(temp_file_path)

        # Run inference with multi-scale fusion & CLAHE contrast boost
        annotated_img, raw_dets = inferrer.predict_image(
            img_rgb,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_tiling=use_tiling,
            enhance_contrast=enhance_contrast,
            multi_scale_fusion=multi_scale_fusion
        )

        # Process and enrich detections
        enriched_dets: List[Dict[str, Any]] = []
        for idx, d in enumerate(raw_dets):
            det_id = idx + 1
            poly = d["poly"]
            obb_info = compute_obb_attributes(poly)
            cx, cy = obb_info["center_px"]

            lat, lon = None, None
            geo_poly = None
            if geo_engine is not None and geo_engine.is_georeferenced:
                lat, lon = geo_engine.pixel_to_latlon(cx, cy)
                geo_poly = geo_engine.polygon_pixels_to_latlon(poly)

            det_entry = {
                "id": det_id,
                "class_id": d["cls_id"],
                "class_name": d["cls_name"],
                "confidence": round(float(d["conf"]), 4),
                "confidence_percent": f"{float(d['conf']) * 100:.1f}%",
                "latitude": lat,
                "longitude": lon,
                "obb": {
                    "center_px": [cx, cy],
                    "width_px": obb_info["width_px"],
                    "height_px": obb_info["height_px"],
                    "angle_degrees": obb_info["angle_deg"],
                    "bbox_xywh_px": obb_info["bbox_px"],
                    "polygon_corners_px": obb_info["polygon_px"],
                    "polygon_corners_latlon": geo_poly
                }
            }

            if reference_lat is not None and reference_lon is not None and lat is not None:
                dist_m = calculate_geo_distance(reference_lat, reference_lon, lat, lon)
                det_entry["distance_from_reference_meters"] = round(dist_m, 2) if dist_m is not None else None
                det_entry["distance_from_reference_formatted"] = format_distance(dist_m)

            enriched_dets.append(det_entry)

        # Build class breakdown
        from collections import Counter
        class_counts = dict(Counter(d["class_name"] for d in enriched_dets))
        confs = [d["confidence"] * 100 for d in enriched_dets]
        avg_conf = round(float(np.mean(confs)), 1) if confs else 0.0
        max_conf = round(float(np.max(confs)), 1) if confs else 0.0

        summary = {
            "total_detections": len(enriched_dets),
            "class_counts": class_counts,
            "resolution": f"{w} × {h} px",
            "format": fmt,
            "average_confidence": f"{avg_conf}%",
            "max_confidence": f"{max_conf}%",
            "is_georeferenced": geo_engine.is_georeferenced if geo_engine else False,
            "crs": geo_engine.crs_str if geo_engine else "Standard Pixel Coordinates"
        }

        # Format detection context for LLM prompt
        formatted_context = self._build_markdown_context(summary, enriched_dets, w, h)

        # Generate base64 data URLs for frontend rendering
        annotated_b64 = self.encode_image_to_data_url(annotated_img)
        preview_b64 = self.encode_image_to_data_url(img_rgb)

        return {
            "annotated_image": annotated_b64,
            "preview_image": preview_b64,
            "detections": enriched_dets,
            "summary": summary,
            "formatted_context": formatted_context
        }

    def _build_markdown_context(
        self,
        summary: Dict[str, Any],
        detections: List[Dict[str, Any]],
        w: int,
        h: int
    ) -> str:
        """Constructs a clean, structured context string for the LLM prompt."""
        lines = []
        lines.append("### 🛰️ Real-Time Satellite / Aerial Object Detection Telemetry")
        lines.append(f"- **Image Dimensions:** {w} × {h} pixels")
        lines.append(f"- **Coordinate System (CRS):** {summary.get('crs', 'Pixel Space')}")
        lines.append(f"- **Total Objects Detected:** {summary.get('total_detections', 0)}")
        
        if summary.get("class_counts"):
            counts_str = ", ".join([f"{cls}: {cnt}" for cls, cnt in sorted(summary["class_counts"].items(), key=lambda x: x[1], reverse=True)])
            lines.append(f"- **Detected Object Breakdown:** {counts_str}")

        if detections:
            lines.append("\n**Key Detections Table:**")
            lines.append("| # | Class | Confidence | Center (px) | Geographic Coords (Lat, Lon) | Dimensions (W×H px) |")
            lines.append("|---|---|---|---|---|---|")
            for d in detections[:25]:  # include up to 25 items for token efficiency
                lat_lon_str = f"({d['latitude']:.6f}, {d['longitude']:.6f})" if d.get("latitude") is not None else "N/A"
                cx, cy = d["obb"]["center_px"]
                dims = f"{d['obb']['width_px']}×{d['obb']['height_px']} px"
                lines.append(f"| {d['id']} | **{d['class_name']}** | {d['confidence_percent']} | ({int(cx)}, {int(cy)}) | {lat_lon_str} | {dims} |")

            if len(detections) > 25:
                lines.append(f"*... and {len(detections) - 25} additional detected objects.*")
        else:
            lines.append("- *No objects exceeded the detection confidence threshold.*")

        return "\n".join(lines)

    def format_detection_context(self, detection_data: Dict[str, Any]) -> str:
        """Constructs a clean, structured context string from stored detection data."""
        if not detection_data:
            return ""
        
        summary = detection_data.get("summary", detection_data)
        detections = detection_data.get("detections", summary.get("detections", []))
        
        res_str = summary.get("resolution", "")
        w, h = 0, 0
        if res_str and "×" in res_str:
            try:
                parts = res_str.split("×")
                w = int(parts[0].strip())
                h = int(parts[1].replace("px", "").strip())
            except Exception:
                pass

        return self._build_markdown_context(summary, detections, w, h)


detection_service = DetectionService()
