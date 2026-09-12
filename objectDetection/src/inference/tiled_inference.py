import os
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

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
        return inter / union
    except Exception:
        return 0.0

def obb_nms(detections, iou_threshold=0.4):
    """
    Non-Maximum Suppression for Oriented Bounding Boxes.
    detections: list of dicts with keys: 'poly' (4x2 np.array), 'conf', 'cls_id', 'cls_name'
    """
    if len(detections) == 0:
        return []
        
    # Group by class id
    by_class = {}
    for d in detections:
        cid = d['cls_id']
        by_class.setdefault(cid, []).append(d)
        
    keep_all = []
    for cid, items in by_class.items():
        # Sort by confidence descending
        items.sort(key=lambda x: x['conf'], reverse=True)
        keep = []
        
        while len(items) > 0:
            best = items.pop(0)
            keep.append(best)
            
            remaining = []
            for other in items:
                iou = polygon_iou(best['poly'], other['poly'])
                if iou < iou_threshold:
                    remaining.append(other)
            items = remaining
            
        keep_all.extend(keep)
        
    return keep_all

class TiledOBBInferrer:
    def __init__(self, model_path="models/best.pt", device=None, tile_size=640, overlap=0.2):
        self.model = YOLO(model_path)
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        
    def predict_image(self, img_rgb, conf_threshold=0.25, iou_threshold=0.45, use_tiling=True):
        """
        Run inference on image (numpy RGB uint8 array).
        Returns:
            annotated_img (RGB uint8),
            detections (list of dicts)
        """
        h, w = img_rgb.shape[:2]
        
        # If image is small or tiling disabled, run standard inference
        if not use_tiling or (h <= self.tile_size and w <= self.tile_size):
            results = self.model.predict(img_rgb, conf=conf_threshold, iou=iou_threshold,
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

        # Slicing image with overlap
        stride = int(self.tile_size * (1 - self.overlap))
        x_starts = list(range(0, max(1, w - self.tile_size + 1), stride))
        if x_starts[-1] + self.tile_size < w:
            x_starts.append(w - self.tile_size)
            
        y_starts = list(range(0, max(1, h - self.tile_size + 1), stride))
        if y_starts[-1] + self.tile_size < h:
            y_starts.append(h - self.tile_size)
            
        all_raw_dets = []
        
        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(x0 + self.tile_size, w)
                y1 = min(y0 + self.tile_size, h)
                tile = img_rgb[y0:y1, x0:x1]
                
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
                        # Shift to global coordinates
                        global_poly = local_poly + np.array([x0, y0])
                        all_raw_dets.append({
                            "cls_id": cid,
                            "cls_name": cname,
                            "conf": float(confs[i]),
                            "poly": global_poly
                        })
                        
        # Apply global OBB NMS
        final_dets = obb_nms(all_raw_dets, iou_threshold=iou_threshold)
        
        # Draw on global image
        annotated_img = img_rgb.copy()
        
        # Unique color per class
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(len(self.model.names), 3)).tolist()
        
        for det in final_dets:
            cid = det['cls_id']
            cname = det['cls_name']
            conf = det['conf']
            poly = det['poly'].astype(np.int32)
            color = colors[cid % len(colors)]
            
            cv2.polylines(annotated_img, [poly.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
            label = f"{cname} {conf:.2f}"
            pt1 = (int(poly[0][0]), max(15, int(poly[0][1]) - 5))
            cv2.putText(annotated_img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
        return annotated_img, final_dets
