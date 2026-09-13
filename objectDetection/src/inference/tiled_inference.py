import os
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

def enhance_satellite_contrast(img_rgb, clip_limit=2.0):
    """
    Applies adaptive histogram equalization (CLAHE) in LAB color space
    to dramatically improve contrast in hazy, low-dynamic-range, or shadow regions.
    """
    try:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enh = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return img_rgb

def polygon_iou(poly1_pts, poly2_pts):
    """Compute IoU between two 4-point polygons."""
    try:
        p1 = Polygon(poly1_pts)
        p2 = Polygon(poly2_pts)
        if not p1.is_valid or not p2.is_valid:
            p1 = p1.buffer(0)
            p2 = p2.buffer(0)
        if not p1.intersects(p2):
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0

def weighted_obb_fusion(detections, iou_threshold=0.45, min_area=12.0):
    """
    Weighted Polygon / Box Fusion for Oriented Bounding Boxes.
    Merges overlapping detections from multiple tiles and scales by computing a
    confidence-weighted average of polygon corners and calibrated confidence boost.
    """
    if len(detections) == 0:
        return []
        
    by_class = {}
    for d in detections:
        # Area check to filter degenerate/micro-pixel noise
        poly = np.array(d['poly'], dtype=np.float32)
        try:
            p = Polygon(poly)
            if p.area < min_area:
                continue
        except Exception:
            continue
        by_class.setdefault(d['cls_id'], []).append(d)
        
    keep_all = []
    for cid, items in by_class.items():
        items.sort(key=lambda x: x['conf'], reverse=True)
        clusters = []
        
        for item in items:
            poly = np.array(item['poly'], dtype=np.float32)
            matched = False
            for cluster in clusters:
                avg_poly = cluster['fused_poly']
                iou = polygon_iou(poly, avg_poly)
                if iou >= iou_threshold:
                    cluster['items'].append(item)
                    # Recompute confidence-weighted polygon
                    weights = np.array([x['conf'] for x in cluster['items']], dtype=np.float32)
                    weights = weights / np.sum(weights)
                    all_polys = np.array([x['poly'] for x in cluster['items']], dtype=np.float32)
                    cluster['fused_poly'] = np.sum(all_polys * weights[:, None, None], axis=0)
                    
                    # Boost confidence slightly when multiple tiles corroborate detection
                    top_conf = max(x['conf'] for x in cluster['items'])
                    cluster['conf'] = min(1.0, float(top_conf + 0.02 * (len(cluster['items']) - 1)))
                    matched = True
                    break
                    
            if not matched:
                clusters.append({
                    'cls_id': cid,
                    'cls_name': item['cls_name'],
                    'conf': item['conf'],
                    'fused_poly': poly,
                    'items': [item]
                })
                
        for c in clusters:
            keep_all.append({
                'cls_id': c['cls_id'],
                'cls_name': c['cls_name'],
                'conf': round(float(c['conf']), 4),
                'poly': c['fused_poly']
            })
            
    # Sort final detections by confidence descending
    keep_all.sort(key=lambda x: x['conf'], reverse=True)
    return keep_all

class TiledOBBInferrer:
    def __init__(self, model_path="models/best.pt", device=None, tile_size=640, overlap=0.25):
        self.model = YOLO(model_path)
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        
    def predict_image(
        self,
        img_rgb,
        conf_threshold=0.20,
        iou_threshold=0.45,
        use_tiling=True,
        enhance_contrast=True,
        multi_scale_fusion=True
    ):
        """
        Run high-accuracy inference on image (numpy RGB uint8 array).
        Features:
          - Adaptive CLAHE contrast enhancement
          - Multi-scale fusion (global contextual view + tiled sliding windows)
          - Weighted Box / Polygon Fusion (WBF-OBB)
        Returns:
            annotated_img (RGB uint8),
            detections (list of dicts)
        """
        h, w = img_rgb.shape[:2]
        
        # Preprocessing: Contrast enhancement
        proc_img = enhance_satellite_contrast(img_rgb, clip_limit=2.0) if enhance_contrast else img_rgb
        
        # If image is small or tiling disabled, run standard inference
        if not use_tiling or (h <= self.tile_size and w <= self.tile_size):
            results = self.model.predict(proc_img, conf=conf_threshold, iou=iou_threshold,
                                         device=self.device, verbose=False)
            detections = []
            res = results[0]
            if res.obb is not None and len(res.obb) > 0:
                xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
                confs = res.obb.conf.cpu().numpy()
                clss = res.obb.cls.cpu().numpy().astype(int)
                
                for i in range(len(clss)):
                    cid = int(clss[i])
                    cname = res.names.get(cid, str(cid))
                    poly = xyxyxyxy[i]
                    detections.append({
                        "cls_id": cid,
                        "cls_name": cname,
                        "conf": float(confs[i]),
                        "poly": poly
                    })
            annotated_img = res.plot()
            if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            return annotated_img, detections

        all_raw_dets = []

        # 1. Global Multi-Scale Context Pass (catches large bridges, aircraft, fields, ships)
        if multi_scale_fusion and (h > self.tile_size or w > self.tile_size):
            res_global = self.model.predict(proc_img, conf=conf_threshold, device=self.device, verbose=False)[0]
            if res_global.obb is not None and len(res_global.obb) > 0:
                xyxyxyxy = res_global.obb.xyxyxyxy.cpu().numpy()
                confs = res_global.obb.conf.cpu().numpy()
                clss = res_global.obb.cls.cpu().numpy().astype(int)
                for i in range(len(clss)):
                    cid = int(clss[i])
                    cname = res_global.names.get(cid, str(cid))
                    all_raw_dets.append({
                        "cls_id": cid,
                        "cls_name": cname,
                        "conf": float(confs[i]),
                        "poly": xyxyxyxy[i]
                    })

        # 2. Sliced Tiled Windows with Overlap (catches small fine-grained vehicles, boats, planes)
        stride = int(self.tile_size * (1 - self.overlap))
        x_starts = list(range(0, max(1, w - self.tile_size + 1), stride))
        if x_starts[-1] + self.tile_size < w:
            x_starts.append(w - self.tile_size)
            
        y_starts = list(range(0, max(1, h - self.tile_size + 1), stride))
        if y_starts[-1] + self.tile_size < h:
            y_starts.append(h - self.tile_size)
            
        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(x0 + self.tile_size, w)
                y1 = min(y0 + self.tile_size, h)
                tile = proc_img[y0:y1, x0:x1]
                
                results = self.model.predict(tile, conf=conf_threshold, device=self.device, verbose=False)
                res = results[0]
                if res.obb is not None and len(res.obb) > 0:
                    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
                    confs = res.obb.conf.cpu().numpy()
                    clss = res.obb.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(clss)):
                        cid = int(clss[i])
                        cname = res.names.get(cid, str(cid))
                        local_poly = xyxyxyxy[i]
                        # Shift to global image coordinates
                        global_poly = local_poly + np.array([x0, y0])
                        all_raw_dets.append({
                            "cls_id": cid,
                            "cls_name": cname,
                            "conf": float(confs[i]),
                            "poly": global_poly
                        })
                        
        # 3. Apply High-Precision Weighted OBB Fusion (WBF)
        final_dets = weighted_obb_fusion(all_raw_dets, iou_threshold=iou_threshold)
        
        # 4. Draw high-fidelity annotations on original image
        annotated_img = img_rgb.copy()
        
        # Consistent curated color palette
        palette = [
            (59, 130, 246), (16, 185, 129), (245, 158, 11), (239, 68, 68),
            (139, 92, 246), (236, 72, 153), (6, 182, 212), (34, 197, 94),
            (249, 115, 22), (168, 85, 247), (20, 184, 166), (244, 63, 94)
        ]
        
        for det in final_dets:
            cid = det['cls_id']
            cname = det['cls_name']
            conf = det['conf']
            poly = det['poly'].astype(np.int32)
            color = palette[cid % len(palette)]
            
            # Draw oriented polygon boundary
            cv2.polylines(annotated_img, [poly.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
            
            # Label background badge
            label = f"{cname} {conf:.2f}"
            pt1 = (int(poly[0][0]), max(16, int(poly[0][1]) - 6))
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated_img, (pt1[0] - 2, pt1[1] - text_h - 2), (pt1[0] + text_w + 2, pt1[1] + baseline), (0, 0, 0), -1)
            cv2.putText(annotated_img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            
        return annotated_img, final_dets

