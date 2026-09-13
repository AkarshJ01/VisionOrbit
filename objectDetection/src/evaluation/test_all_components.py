import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ultralytics import YOLO
from src.inference.tiled_inference import TiledOBBInferrer, polygon_iou, obb_nms
from src.inference.predict import predict
from src.inference.georeference import GeoReferenceEngine, calculate_geo_distance

class TestSatQueryPipeline(unittest.TestCase):
    def setUp(self):
        self.model_path = "models/best.pt"
        self.data_yaml = "data/fair1m_obb.yaml"
        self.test_img_dir = "data/processed/images/test"
        
    def test_01_environment(self):
        """Verify PyTorch and MPS/CPU backend."""
        self.assertTrue(hasattr(torch, '__version__'))
        print(f"\n[PASS] PyTorch Version: {torch.__version__}, MPS Available: {torch.backends.mps.is_available()}")

    def test_02_model_weights(self):
        """Verify model file exists and is valid YOLO-OBB."""
        self.assertTrue(os.path.exists(self.model_path), f"Model not found at {self.model_path}")
        model = YOLO(self.model_path)
        self.assertEqual(model.task, "obb")
        self.assertEqual(len(model.names), 37)
        print(f"[PASS] YOLO-OBB Model loaded successfully with {len(model.names)} classes.")

    def test_03_polygon_iou_and_nms(self):
        """Test Shapely-based polygon IoU and OBB NMS."""
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        poly2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        iou_ident = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou_ident, 1.0, places=3)
        
        poly3 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]])
        iou_disjoint = polygon_iou(poly1, poly3)
        self.assertAlmostEqual(iou_disjoint, 0.0, places=3)
        
        # Test NMS
        dets = [
            {"cls_id": 0, "cls_name": "Small Car", "conf": 0.9, "poly": poly1},
            {"cls_id": 0, "cls_name": "Small Car", "conf": 0.8, "poly": poly2}, # duplicate
            {"cls_id": 0, "cls_name": "Small Car", "conf": 0.7, "poly": poly3}, # distinct
        ]
        kept = obb_nms(dets, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)
        print("[PASS] Polygon IoU and OBB NMS logic verified.")

    def test_04_tiled_inference(self):
        """Test high-resolution tiled inference."""
        inferrer = TiledOBBInferrer(model_path=self.model_path, tile_size=640, overlap=0.2)
        # Create synthetic large image (1280x1280)
        dummy_large = np.zeros((1280, 1280, 3), dtype=np.uint8)
        annotated, dets = inferrer.predict_image(dummy_large, conf_threshold=0.25, use_tiling=True)
        self.assertEqual(annotated.shape, (1280, 1280, 3))
        self.assertIsInstance(dets, list)
        print("[PASS] Tiled inference executed on 1280x1280 canvas.")

    def test_05_predict_cli_on_test_sample(self):
        """Test inference CLI execution on a real test sample."""
        if os.path.exists(self.test_img_dir):
            sample_files = [f for f in os.listdir(self.test_img_dir) if f.endswith(('.jpg', '.png', '.tif'))]
            if sample_files:
                sample_path = os.path.join(self.test_img_dir, sample_files[0])
                results = predict(sample_path, model_path=self.model_path, conf=0.15, save_dir="outputs/predictions")
                self.assertGreaterEqual(len(results), 1)
                self.assertTrue(os.path.exists(os.path.join("outputs/predictions", f"pred_{os.path.splitext(sample_files[0])[0]}.jpg")))
                self.assertTrue(os.path.exists(os.path.join("outputs/predictions", f"pred_{os.path.splitext(sample_files[0])[0]}.json")))
                print(f"[PASS] CLI inference verified on sample: {sample_path}")

    def test_06_georeferencing_and_distances(self):
        """Test GeoTIFF metadata extraction and WGS84 distance calculation."""
        tif_path = os.path.expanduser("~/1041.tif")
        if os.path.exists(tif_path):
            geo = GeoReferenceEngine(tif_path)
            self.assertTrue(geo.is_georeferenced)
            self.assertEqual(geo.crs_str, "EPSG:4326")
            lat, lon = geo.pixel_to_latlon(500, 500)
            self.assertIsNotNone(lat)
            self.assertIsNotNone(lon)
            
            # Test geodesic distance calculation
            dist = calculate_geo_distance(lat, lon, 40.108, -88.239)
            self.assertGreater(dist, 0.0)
            print(f"[PASS] GeoTIFF georeferencing & geodesic distance verified ({dist:.1f} m).")

if __name__ == "__main__":
    unittest.main(verbosity=2)
