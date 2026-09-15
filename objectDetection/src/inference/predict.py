import os
import sys
import argparse
import json
import cv2
import numpy as np
import tifffile
from PIL import Image
from collections import Counter


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.inference.tiled_inference import TiledOBBInferrer
from src.inference.georeference import (
    GeoReferenceEngine,
    calculate_geo_distance,
    format_distance
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "optical": "models/best.pt",
    "sar": "models/best_model.pth"
}


# ============================================================
# NPY LOADING
# ============================================================

def load_npy_rgb(path):
    """
    Load a NumPy .npy satellite/SAR image.

    Supported layouts:

        H x W
        H x W x C
        C x H x W

    The result is converted into an RGB uint8 image so that
    the existing OBB inference pipeline can consume it.

    IMPORTANT:
    This assumes the SAR model expects an image-like array
    after conversion. If best_model.pth expects raw tensors
    with a specific number of SAR channels, the SAR loader
    must be adapted accordingly.
    """

    arr = np.load(path, allow_pickle=False)

    if not isinstance(arr, np.ndarray):
        raise ValueError(".npy file did not contain a NumPy array")

    if arr.size == 0:
        raise ValueError(".npy file contains an empty array")

    print(f"[SAR] Loaded NPY shape: {arr.shape}")
    print(f"[SAR] NPY dtype: {arr.dtype}")

    # --------------------------------------------------------
    # H x W
    # --------------------------------------------------------

    if arr.ndim == 2:

        img = arr

        img = np.nan_to_num(
            img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:

            img = (
                (img - img_min)
                / (img_max - img_min)
                * 255.0
            )

        else:

            img = np.zeros_like(
                img,
                dtype=np.float32
            )

        img = img.astype(np.uint8)

        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    # --------------------------------------------------------
    # H x W x C
    # --------------------------------------------------------

    if arr.ndim == 3:

        # Normal image-style layout
        if arr.shape[2] <= 16:

            img = arr

        # ----------------------------------------------------
        # C x H x W
        # ----------------------------------------------------

        elif arr.shape[0] <= 16:

            img = np.transpose(
                arr,
                (1, 2, 0)
            )

        else:

            raise ValueError(
                f"Unsupported NPY shape: {arr.shape}"
            )

        img = np.nan_to_num(
            img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Single channel
        # ----------------------------------------------------

        if img.shape[2] == 1:

            channel = img[:, :, 0]

            cmin = np.min(channel)
            cmax = np.max(channel)

            if cmax > cmin:

                channel = (
                    (channel - cmin)
                    / (cmax - cmin)
                    * 255.0
                )

            else:

                channel = np.zeros_like(
                    channel,
                    dtype=np.float32
                )

            channel = channel.astype(np.uint8)

            return cv2.cvtColor(
                channel,
                cv2.COLOR_GRAY2RGB
            )

        # ----------------------------------------------------
        # Two-channel SAR
        #
        # Example:
        # channel 0 = VV
        # channel 1 = VH
        #
        # We create a 3-channel representation:
        #
        # R = channel 0
        # G = channel 1
        # B = mean(channel 0, channel 1)
        # ----------------------------------------------------

        if img.shape[2] == 2:

            channels = []

            for i in range(2):

                channel = img[:, :, i]

                cmin = np.min(channel)
                cmax = np.max(channel)

                if cmax > cmin:

                    channel = (
                        (channel - cmin)
                        / (cmax - cmin)
                        * 255.0
                    )

                else:

                    channel = np.zeros_like(
                        channel,
                        dtype=np.float32
                    )

                channels.append(
                    channel.astype(np.uint8)
                )

            combined = np.mean(
                np.stack(channels, axis=2),
                axis=2
            ).astype(np.uint8)

            return np.stack(
                [
                    channels[0],
                    channels[1],
                    combined
                ],
                axis=2
            )

        # ----------------------------------------------------
        # Three or more channels
        # ----------------------------------------------------

        img = img[:, :, :3]

        output_channels = []

        for i in range(3):

            channel = img[:, :, i]

            cmin = np.min(channel)
            cmax = np.max(channel)

            if cmax > cmin:

                channel = (
                    (channel - cmin)
                    / (cmax - cmin)
                    * 255.0
                )

            else:

                channel = np.zeros_like(
                    channel,
                    dtype=np.float32
                )

            output_channels.append(
                channel.astype(np.uint8)
            )

        return np.stack(
            output_channels,
            axis=2
        )

    raise ValueError(
        f"Unsupported NPY dimensions: {arr.ndim}"
    )


# ============================================================
# GENERAL IMAGE LOADING
# ============================================================

def load_image_rgb(path):

    path = os.path.expanduser(path)

    # --------------------------------------------------------
    # NPY → SAR
    # --------------------------------------------------------

    if path.lower().endswith(".npy"):

        return load_npy_rgb(path)

    # --------------------------------------------------------
    # TIFF
    # --------------------------------------------------------

    if path.lower().endswith((".tif", ".tiff")):

        try:

            img = tifffile.imread(path)

            if img is None:
                raise ValueError(
                    "TIFF image could not be read"
                )

            if len(img.shape) == 2:

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_GRAY2RGB
                )

            elif len(img.shape) == 3:

                if img.shape[2] >= 3:

                    img = img[:, :, :3]

                elif img.shape[2] == 1:

                    img = cv2.cvtColor(
                        img[:, :, 0],
                        cv2.COLOR_GRAY2RGB
                    )

            if img.dtype != np.uint8:

                img_min = np.nanmin(img)
                img_max = np.nanmax(img)

                if img_max > img_min:

                    img = (
                        (img - img_min)
                        / (img_max - img_min)
                        * 255
                    ).astype(np.uint8)

                else:

                    img = np.zeros(
                        img.shape,
                        dtype=np.uint8
                    )

            return img

        except Exception as e:

            print(
                f"TIFF reader failed for {path}: {e}"
            )

    # --------------------------------------------------------
    # OpenCV
    # --------------------------------------------------------

    img = cv2.imread(path)

    if img is not None:

        return cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    # --------------------------------------------------------
    # PIL
    # --------------------------------------------------------

    try:

        with Image.open(path) as pimg:

            return np.array(
                pimg.convert("RGB")
            )

    except Exception as e:

        raise ValueError(
            f"Could not read image: {path}\n{e}"
        )


# ============================================================
# OBB ATTRIBUTES
# ============================================================

def compute_obb_attributes(poly_pts):

    pts = np.array(
        poly_pts,
        dtype=np.float32
    )

    if pts.shape[0] < 4:

        raise ValueError(
            "OBB polygon must contain at least 4 points"
        )

    cx = float(
        np.mean(pts[:, 0])
    )

    cy = float(
        np.mean(pts[:, 1])
    )

    e0 = np.linalg.norm(
        pts[1] - pts[0]
    )

    e1 = np.linalg.norm(
        pts[2] - pts[1]
    )

    width = float(
        max(e0, e1)
    )

    height = float(
        min(e0, e1)
    )

    if e0 >= e1:

        vec = pts[1] - pts[0]

    else:

        vec = pts[2] - pts[1]

    angle_deg = float(
        np.degrees(
            np.arctan2(
                vec[1],
                vec[0]
            )
        )
    )

    min_x = float(
        np.min(pts[:, 0])
    )

    min_y = float(
        np.min(pts[:, 1])
    )

    max_x = float(
        np.max(pts[:, 0])
    )

    max_y = float(
        np.max(pts[:, 1])
    )

    return {

        "center_px": [
            round(cx, 1),
            round(cy, 1)
        ],

        "width_px":
            round(width, 1),

        "height_px":
            round(height, 1),

        "angle_deg":
            round(angle_deg, 2),

        "bbox_px": [

            round(min_x, 1),
            round(min_y, 1),
            round(max_x - min_x, 1),
            round(max_y - min_y, 1)

        ],

        "polygon_px": [

            [
                round(float(p[0]), 1),
                round(float(p[1]), 1)
            ]

            for p in pts

        ]
    }


# ============================================================
# MODEL TYPE
# ============================================================

def detect_model_type(source, model_path=None, prompt=None):

    if model_path is not None:

        if model_path == MODEL_PATHS["sar"]:

            return "sar"

        if model_path == MODEL_PATHS["optical"]:

            return "optical"

        return "custom"

    # File type has priority
    if str(source).lower().endswith(".npy"):

        return "sar"

    # Prompt fallback
    if prompt:

        prompt_lower = prompt.lower()

        sar_keywords = [
            "sar",
            "synthetic aperture radar",
            "radar image",
            "radar imagery",
            "sentinel-1",
            "sentinel 1",
            "sentinel1",
            "sar image",
            "sar imagery",
            "sar data",
            "radar data",
            "microwave image",
            "microwave imagery"
        ]

        for keyword in sar_keywords:

            if keyword in prompt_lower:

                return "sar"

    return "optical"


# ============================================================
# DETECTION REPORT
# ============================================================

def print_detection_report(
    image_name,
    img_w,
    img_h,
    detections,
    model_name="YOLO11n-OBB",
    crs_str=None
):

    print("\n" + "=" * 80)
    print("              SATELLITE OBJECT DETECTION")
    print("=" * 80)

    print(f"Image:       {image_name}")
    print(f"Resolution:  {img_w} × {img_h}")
    print(f"Model:       {model_name}")

    print(
        f"CRS:         "
        f"{crs_str if crs_str else 'N/A'}"
    )

    print(
        f"Detections:  {len(detections)}"
    )

    print("-" * 80)

    if len(detections) == 0:

        print("No objects detected.")

    else:

        print(
            f"{'#':<4}"
            f"{'CLASS':<20}"
            f"{'CONFIDENCE':<12}"
            f"{'CENTER(px)':<16}"
            f"{'LATITUDE':<14}"
            f"{'LONGITUDE':<14}"
        )

        print("-" * 80)

        for d in detections:

            cx, cy = d["obb"]["center_px"]

            lat = (
                f"{d['latitude']:.6f}"
                if d["latitude"] is not None
                else "N/A"
            )

            lon = (
                f"{d['longitude']:.6f}"
                if d["longitude"] is not None
                else "N/A"
            )

            print(
                f"{d['id']:<4}"
                f"{d['class_name']:<20}"
                f"{d['confidence'] * 100:.1f}%     "
                f"({int(cx)},{int(cy)})     "
                f"{lat:<14}"
                f"{lon:<14}"
            )

    print("=" * 80)


# ============================================================
# PREDICTION
# ============================================================

def predict(
    source,
    model_path=None,
    prompt=None,
    conf=0.20,
    iou=0.45,
    tiled=True,
    tile_size=640,
    enhance_contrast=True,
    multi_scale_fusion=True,
    save_dir="outputs/predictions"
):

    source = os.path.expanduser(source)

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ========================================================
    # MODEL ROUTING
    # ========================================================

    model_type = detect_model_type(
        source,
        model_path,
        prompt
    )

    if model_path is None:

        model_path = MODEL_PATHS[
            model_type
        ]

    print("\n" + "=" * 80)
    print("                         MODEL ROUTING")
    print("=" * 80)

    print(f"Source:       {source}")
    print(f"Model type:   {model_type}")
    print(f"Model path:   {model_path}")

    if model_type == "sar":

        print(
            "[ROUTER] .npy/SAR input detected."
        )

        print(
            "[ROUTER] Running SAR inference directly."
        )

    print("=" * 80)

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # ========================================================
    # INFERRER
    # ========================================================

    inferrer = TiledOBBInferrer(
        model_path=model_path,
        tile_size=tile_size
    )

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    supported_extensions = (
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png",
        ".npy"
    )

    if os.path.isfile(source):

        files = [source]

    elif os.path.isdir(source):

        files = [

            os.path.join(
                source,
                f
            )

            for f in sorted(
                os.listdir(source)
            )

            if f.lower().endswith(
                supported_extensions
            )
        ]

    else:

        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if not files:

        raise FileNotFoundError(
            f"No supported files found in {source}"
        )

    results_summary = []

    # ========================================================
    # PROCESS
    # ========================================================

    for fpath in files:

        fname = os.path.basename(fpath)

        stem = os.path.splitext(fname)[0]

        print("\n" + "-" * 80)
        print(f"Processing: {fname}")
        print("-" * 80)

        # ====================================================
        # LOAD
        # ====================================================

        try:

            img_rgb = load_image_rgb(
                fpath
            )

        except Exception as e:

            print(
                f"[ERROR] Could not load {fname}: {e}"
            )

            continue

        h, w = img_rgb.shape[:2]

        # ====================================================
        # GEOREFERENCE
        # ====================================================

        geo_engine = None

        try:

            geo_engine = GeoReferenceEngine(
                fpath
            )

        except Exception as e:

            print(
                f"[GEO] Georeferencing unavailable: {e}"
            )

        # ====================================================
        # INFERENCE
        # ====================================================

        try:

            annotated_img, raw_dets = (
                inferrer.predict_image(

                    img_rgb,

                    conf_threshold=conf,

                    iou_threshold=iou,

                    use_tiling=tiled,

                    enhance_contrast=
                        enhance_contrast,

                    multi_scale_fusion=
                        multi_scale_fusion

                )
            )

        except Exception as e:

            print(
                f"[ERROR] Inference failed: {e}"
            )

            continue

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        out_img_path = os.path.join(
            save_dir,
            f"pred_{stem}.jpg"
        )

        Image.fromarray(
            annotated_img
        ).save(
            out_img_path,
            quality=95
        )

        # ====================================================
        # ENRICH DETECTIONS
        # ====================================================

        enriched_dets = []

        for idx, d in enumerate(raw_dets):

            det_id = idx + 1

            poly = d["poly"]

            obb = compute_obb_attributes(
                poly
            )

            cx, cy = obb["center_px"]

            lat = None
            lon = None
            geo_poly = None

            if geo_engine is not None:

                try:

                    lat, lon = (
                        geo_engine.pixel_to_latlon(
                            cx,
                            cy
                        )
                    )

                    geo_poly = (
                        geo_engine
                        .polygon_pixels_to_latlon(
                            poly
                        )
                    )

                except Exception:

                    pass

            enriched_dets.append({

                "id":
                    det_id,

                "class_id":
                    d["cls_id"],

                "class_name":
                    d["cls_name"],

                "confidence":
                    round(
                        float(d["conf"]),
                        4
                    ),

                "latitude":
                    lat,

                "longitude":
                    lon,

                "obb": {

                    "center_px":
                        obb["center_px"],

                    "width_px":
                        obb["width_px"],

                    "height_px":
                        obb["height_px"],

                    "angle_degrees":
                        obb["angle_deg"],

                    "bbox_xywh_px":
                        obb["bbox_px"],

                    "polygon_corners_px":
                        obb["polygon_px"],

                    "polygon_corners_latlon":
                        geo_poly
                }
            })

        # ====================================================
        # MODEL NAME
        # ====================================================

        if model_type == "sar":

            model_display_name = (
                "YOLO11n-OBB-SAR"
            )

        elif model_type == "optical":

            model_display_name = (
                "YOLO11n-OBB"
            )

        else:

            model_display_name = (
                "Custom OBB Model"
            )

        # ====================================================
        # REPORT
        # ====================================================

        print_detection_report(
            image_name=fname,
            img_w=w,
            img_h=h,
            detections=enriched_dets,
            model_name=model_display_name,
            crs_str=(
                geo_engine.crs_str
                if geo_engine is not None
                else None
            )
        )

        # ====================================================
        # JSON
        # ====================================================

        output = {

            "source_file":
                fpath,

            "filename":
                fname,

            "input_type":
                "npy_sar"
                if fname.lower().endswith(".npy")
                else "image",

            "model": {

                "type":
                    model_type,

                "path":
                    model_path

            },

            "image_dimensions": {

                "width":
                    w,

                "height":
                    h,

                "channels":
                    (
                        img_rgb.shape[2]
                        if img_rgb.ndim == 3
                        else 1
                    )
            },

            "georeferencing": {

                "is_georeferenced":
                    (
                        geo_engine.is_georeferenced
                        if geo_engine is not None
                        else False
                    ),

                "crs":
                    (
                        geo_engine.crs_str
                        if geo_engine is not None
                        else None
                    ),

                "bounds":
                    (
                        geo_engine.bounds
                        if geo_engine is not None
                        else None
                    )
            },

            "detection_count":
                len(enriched_dets),

            "detections":
                enriched_dets
        }

        out_json_path = os.path.join(
            save_dir,
            f"pred_{stem}.json"
        )

        with open(
            out_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Annotated Image: {out_img_path}"
        )

        print(
            f"Detection JSON:  {out_json_path}"
        )

        results_summary.append(
            output
        )

    return results_summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
            "VisionOrbit Satellite Object Detection"
    )

    parser.add_argument(
        "--source",
        required=True,
        type=str,
        help=
            "Image or .npy SAR file"
    )

    parser.add_argument(
        "--prompt",
        default=None,
        type=str
    )

    parser.add_argument(
        "--model",
        default=None,
        type=str
    )

    parser.add_argument(
        "--conf",
        default=0.20,
        type=float
    )

    parser.add_argument(
        "--iou",
        default=0.45,
        type=float
    )

    parser.add_argument(
        "--tiled",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--no-tiled",
        action="store_false",
        dest="tiled"
    )

    parser.add_argument(
        "--tile-size",
        default=640,
        type=int
    )

    parser.add_argument(
        "--save-dir",
        default="outputs/predictions",
        type=str
    )

    parser.add_argument(
        "--no-contrast",
        action="store_true"
    )

    parser.add_argument(
        "--no-multiscale",
        action="store_true"
    )

    args = parser.parse_args()

    predict(
        source=args.source,
        model_path=args.model,
        prompt=args.prompt,
        conf=args.conf,
        iou=args.iou,
        tiled=args.tiled,
        tile_size=args.tile_size,
        enhance_contrast=not args.no_contrast,
        multi_scale_fusion=not args.no_multiscale,
        save_dir=args.save_dir
    )