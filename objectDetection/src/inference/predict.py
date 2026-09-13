import os
import sys
import argparse
import json
import math
import cv2
import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.tiled_inference import TiledOBBInferrer
from src.inference.georeference import (
    GeoReferenceEngine,
    calculate_geo_distance,
    format_distance
)

def load_image_rgb(path):
    path = os.path.expanduser(path)
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(path)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            return img
        except Exception as e:
            pass
            
    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    with Image.open(path) as pimg:
        return np.array(pimg.convert("RGB"))

def compute_obb_attributes(poly_pts):
    """
    Computes geometric center, dimensions (width, height), rotation angle, 
    and horizontal bounding box from 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    pts = np.array(poly_pts, dtype=np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    
    # Calculate edge lengths
    e0 = np.linalg.norm(pts[1] - pts[0])
    e1 = np.linalg.norm(pts[2] - pts[1])
    
    # Width is the longer side, height is the shorter side
    width = float(max(e0, e1))
    height = float(min(e0, e1))
    
    # Calculate angle of the longest edge in degrees
    if e0 >= e1:
        vec = pts[1] - pts[0]
    else:
        vec = pts[2] - pts[1]
    angle_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
    
    min_x, min_y = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    max_x, max_y = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    
    return {
        "center_px": (round(cx, 1), round(cy, 1)),
        "width_px": round(width, 1),
        "height_px": round(height, 1),
        "angle_deg": round(angle_deg, 2),
        "bbox_px": [round(min_x, 1), round(min_y, 1), round(round(max_x - min_x, 1), 1), round(round(max_y - min_y, 1), 1)],
        "polygon_px": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts]
    }

def print_detection_report(image_name, img_w, img_h, detections, model_name="YOLO11n-OBB", 
                           crs_str=None, ref_lat=None, ref_lon=None, pairwise=False):
    print("\n" + "=" * 80)
    print("                      SATELLITE OBJECT DETECTION")
    print("=" * 80)
    print(f"Image:       {image_name}")
    print(f"Resolution:  {img_w} × {img_h}")
    print(f"Model:       {model_name}")
    print(f"CRS:         {crs_str if crs_str else 'N/A (Standard Image Coordinates)'}")
    print(f"Detections:  {len(detections)}")
    print("-" * 80)
    
    if len(detections) > 0:
        print(f"{'#':<4} {'CLASS':<20} {'CONFIDENCE':<12} {'CENTER(px)':<16} {'LATITUDE':<14} {'LONGITUDE':<14}")
        print("-" * 80)
        for d in detections:
            det_id = d["id"]
            cname = d["class_name"]
            conf_str = f"{d['confidence']*100:.1f}%"
            cx, cy = d["obb"]["center_px"]
            center_str = f"({int(cx)},{int(cy)})"
            lat_str = f"{d['latitude']:.6f}" if d["latitude"] is not None else "N/A"
            lon_str = f"{d['longitude']:.6f}" if d["longitude"] is not None else "N/A"
            print(f"{det_id:<4} {cname:<20} {conf_str:<12} {center_str:<16} {lat_str:<14} {lon_str:<14}")
    else:
        print("No objects detected at the specified confidence threshold.")
        
    print("\n" + "=" * 80)
    print("                                SUMMARY")
    print("=" * 80)
    print(f"Total detections: {len(detections)}")
    
    if len(detections) > 0:
        from collections import Counter
        class_counts = Counter(d["class_name"] for d in detections)
        print("\nObjects by class:")
        for cname, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cname}: {count}")
            
        confs = [d["confidence"] * 100 for d in detections]
        print(f"\nAverage confidence: {np.mean(confs):.1f}%")
        print(f"Highest confidence: {np.max(confs):.1f}% ({detections[np.argmax(confs)]['class_name']} #{detections[np.argmax(confs)]['id']})")
        print(f"Lowest confidence:  {np.min(confs):.1f}% ({detections[np.argmin(confs)]['class_name']} #{detections[np.argmin(confs)]['id']})")
        
        # Reference point distance reporting
        if ref_lat is not None and ref_lon is not None:
            print("\n" + "-" * 80)
            print(f"DISTANCE FROM REFERENCE POINT ({ref_lat:.6f}, {ref_lon:.6f}):")
            print("Note: Distances represent 2D geographic surface distances on Earth's ellipsoid (WGS84).")
            print("-" * 80)
            for d in detections:
                dist = d.get("distance_from_reference_meters")
                dist_str = format_distance(dist)
                print(f"Detection #{d['id']} ({d['class_name']}): {dist_str}")
                
        # Pairwise distance reporting if requested
        if pairwise and len(detections) > 1 and detections[0]["latitude"] is not None:
            print("\n" + "-" * 80)
            print("KEY PAIRWISE GEOGRAPHIC DISTANCES:")
            print("-" * 80)
            for i in range(min(5, len(detections))):
                for j in range(i + 1, min(5, len(detections))):
                    d1, d2 = detections[i], detections[j]
                    dist = calculate_geo_distance(d1["latitude"], d1["longitude"], d2["latitude"], d2["longitude"])
                    print(f"Distance between #{d1['id']} ({d1['class_name']}) and #{d2['id']} ({d2['class_name']}): {format_distance(dist)}")
                    
    print("=" * 80 + "\n")

def predict(source, model_path="models/best.pt", conf=0.20, iou=0.45, tiled=True, tile_size=640,
            enhance_contrast=True, multi_scale_fusion=True,
            save_dir="outputs/predictions", reference_lat=None, reference_lon=None, pairwise=False):
    
    source = os.path.expanduser(source)
    os.makedirs(save_dir, exist_ok=True)
    inferrer = TiledOBBInferrer(model_path=model_path, tile_size=tile_size)
    
    if os.path.isfile(source):
        files = [source]
    elif os.path.isdir(source):
        files = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
    else:
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
        
    results_summary = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        stem = os.path.splitext(fname)[0]
        
        img_rgb = load_image_rgb(fpath)
        if img_rgb is None:
            print(f"Skipping unreadable image: {fpath}")
            continue
            
        h, w = img_rgb.shape[:2]
        geo_engine = GeoReferenceEngine(fpath)
        
        annotated_img, raw_dets = inferrer.predict_image(
            img_rgb,
            conf_threshold=conf,
            iou_threshold=iou,
            use_tiling=tiled,
            enhance_contrast=enhance_contrast,
            multi_scale_fusion=multi_scale_fusion
        )
        
        # Save annotated image
        out_img_path = os.path.join(save_dir, f"pred_{stem}.jpg")
        Image.fromarray(annotated_img).save(out_img_path, quality=95)
        
        # Process and enrich each detection
        enriched_dets = []
        for idx, d in enumerate(raw_dets):
            det_id = idx + 1
            poly = d["poly"] # 4x2 numpy array
            obb_info = compute_obb_attributes(poly)
            cx, cy = obb_info["center_px"]
            
            # Compute Geolocation
            lat, lon = geo_engine.pixel_to_latlon(cx, cy)
            geo_poly = geo_engine.polygon_pixels_to_latlon(poly)
            
            det_entry = {
                "id": det_id,
                "class_id": d["cls_id"],
                "class_name": d["cls_name"],
                "confidence": round(float(d["conf"]), 4),
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
            
            # Reference coordinate distance
            if reference_lat is not None and reference_lon is not None and lat is not None:
                dist_m = calculate_geo_distance(reference_lat, reference_lon, lat, lon)
                det_entry["distance_from_reference_meters"] = round(dist_m, 2) if dist_m is not None else None
                det_entry["distance_from_reference_formatted"] = format_distance(dist_m)
                
            enriched_dets.append(det_entry)
            
        # Print formatted human-readable terminal report
        print_detection_report(
            image_name=fname,
            img_w=w,
            img_h=h,
            detections=enriched_dets,
            model_name="YOLO11n-OBB",
            crs_str=geo_engine.crs_str,
            ref_lat=reference_lat,
            ref_lon=reference_lon,
            pairwise=pairwise
        )
        
        # Save structured JSON output
        out_json_path = os.path.join(save_dir, f"pred_{stem}.json")
        json_output = {
            "source_image": fpath,
            "filename": fname,
            "image_dimensions": {"width": w, "height": h, "channels": img_rgb.shape[2] if len(img_rgb.shape)==3 else 1},
            "georeferencing": {
                "is_georeferenced": geo_engine.is_georeferenced,
                "crs": geo_engine.crs_str,
                "bounds": geo_engine.bounds
            },
            "detection_count": len(enriched_dets),
            "reference_point": {
                "latitude": reference_lat,
                "longitude": reference_lon
            } if (reference_lat is not None and reference_lon is not None) else None,
            "distance_measurement_note": "Distances represent 2D geographic surface geodesic distance on Earth's ellipsoid (WGS84), distinct from 3D sensor-to-object altitude.",
            "detections": enriched_dets
        }
        
        with open(out_json_path, "w") as jf:
            json.dump(json_output, jf, indent=2)
            
        print(f"Annotated Image: {out_img_path}")
        print(f"Detection JSON:  {out_json_path}")
        
        results_summary.append(json_output)
        
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery Satellite Image Object Detection & Geolocation")
    parser.add_argument("--source", type=str, required=True, help="Path to image (.tif, .png, .jpg) or directory")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO-OBB model weights")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--tiled", action="store_true", default=True, help="Use tiled sliding window inference")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for sliced inference")
    parser.add_argument("--save-dir", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--reference-lat", type=float, default=None, help="Reference latitude for distance calculation")
    parser.add_argument("--reference-lon", type=float, default=None, help="Reference longitude for distance calculation")
    parser.add_argument("--calc-pairwise-dist", action="store_true", help="Calculate pairwise distances between detected objects")
    parser.add_argument("--no-contrast", action="store_true", help="Disable adaptive CLAHE contrast enhancement")
    parser.add_argument("--no-multiscale", action="store_true", help="Disable global multi-scale context fusion")
    
    args = parser.parse_args()
    predict(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        tiled=args.tiled,
        tile_size=args.tile_size,
        enhance_contrast=not args.no_contrast,
        multi_scale_fusion=not args.no_multiscale,
        save_dir=args.save_dir,
        reference_lat=args.reference_lat,
        reference_lon=args.reference_lon,
        pairwise=args.calc_pairwise_dist
    )

