import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from backend.detection_service import detection_service
from backend.geospatial_service import geospatial_service
from backend.multimodal_service import multimodal_service
from backend.change_service import change_service
from backend.query_router import query_router
from backend.report_service import report_service
from SAR.src.inference.sentinel_cnn import run_sentinel_inference, extract_detections


class TestSatQuerySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_tif = "data/raw/images/1.tif"
        cls.sample_tif_0 = "data/raw/images/0.tif"
        cls.sample_npy = "uploads/bfb19c2a6dbd4c83b95007fcd6b01502.npy"

    # ============================================================
    # 1. BASELINE PRESERVATION: 37-CLASS YOLO-OBB DETECTOR
    # ============================================================
    def test_01_yolo_obb_detection_baseline(self):
        if not os.path.exists(self.sample_tif):
            self.skipTest(f"{self.sample_tif} not found")

        res = detection_service.detect(self.sample_tif, conf_threshold=0.20)
        self.assertIn("summary", res)
        self.assertIn("detections", res)
        self.assertIn("annotated_image", res)
        
        summary = res["summary"]
        self.assertGreater(summary["total_detections"], 0, "Expected positive detection count on sample GeoTIFF")
        self.assertTrue(summary["is_georeferenced"], "Expected 1.tif to be georeferenced")
        self.assertEqual(summary["crs"], "EPSG:4326")

        # Check first detection structure
        first_det = res["detections"][0]
        self.assertIn("obb", first_det)
        self.assertIn("center_px", first_det["obb"])
        self.assertIn("latitude", first_det)
        self.assertIn("longitude", first_det)
        print(f"[TEST PASS] YOLO-OBB baseline verified: {summary['total_detections']} detections.")

    # ============================================================
    # 2. BASELINE PRESERVATION: SENTINEL CNN SAR INFERENCE
    # ============================================================
    def test_02_sentinel_cnn_sar_baseline(self):
        if not os.path.exists(self.sample_npy):
            self.skipTest(f"{self.sample_npy} not found")

        pred = run_sentinel_inference(self.sample_npy)
        self.assertIn("mask", pred)
        self.assertEqual(pred["mask"].shape, (256, 256))
        
        regions = extract_detections(pred["mask"])
        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0, "Expected at least 1 detected SAR region")
        print(f"[TEST PASS] Sentinel CNN baseline verified: {len(regions)} regions extracted.")

    # ============================================================
    # 3. SCIENTIFIC SAR .NPY PROCESSING & COLORMAP
    # ============================================================
    def test_03_scientific_sar_processing(self):
        if not os.path.exists(self.sample_npy):
            self.skipTest(f"{self.sample_npy} not found")

        sar_res = multimodal_service.process_sar_npy(
            self.sample_npy,
            colormap_name="inferno",
            speckle_filter=True
        )
        self.assertIn("preview_image", sar_res)
        self.assertIn("composite_image", sar_res)
        self.assertIn("mean_backscatter_db", sar_res)
        self.assertTrue(sar_res["preview_image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(sar_res["shape"][0], 7, "Expected 7 Sentinel channels")
        print(f"[TEST PASS] Scientific SAR processing verified: mean dB = {sar_res['mean_backscatter_db']}.")

    # ============================================================
    # 4. DETERMINISTIC GEOSPATIAL AOI & DISTANCE SERVICE
    # ============================================================
    def test_04_geospatial_service_and_aoi(self):
        # Test geodesic distance
        dist = geospatial_service.calculate_distance(42.280, -71.778, 42.281, -71.778)
        self.assertGreater(dist, 100.0)
        self.assertLess(dist, 120.0)

        # Test AOI point-in-polygon filtering
        dummy_dets = [
            {"id": 1, "class_name": "Small Car", "latitude": 42.2805, "longitude": -71.7789, "obb": {"center_px": [100, 100]}},
            {"id": 2, "class_name": "Van", "latitude": 42.2900, "longitude": -71.7900, "obb": {"center_px": [900, 900]}},
        ]
        aoi_polygon = [
            [42.2800, -71.7800],
            [42.2810, -71.7800],
            [42.2810, -71.7780],
            [42.2800, -71.7780]
        ]
        aoi_res = geospatial_service.filter_detections_in_aoi(dummy_dets, aoi_polygon, is_latlon=True)
        self.assertEqual(aoi_res["total_in_aoi"], 1)
        self.assertEqual(aoi_res["total_outside_aoi"], 1)
        self.assertEqual(aoi_res["matching_detections"][0]["id"], 1)
        print("[TEST PASS] Geospatial AOI Point-In-Polygon verified.")

    # ============================================================
    # 5. TEMPORAL CHANGE DETECTION
    # ============================================================
    def test_05_temporal_change_detection(self):
        if not (os.path.exists(self.sample_tif_0) and os.path.exists(self.sample_tif)):
            self.skipTest("Sample TIFFs for change detection not found")

        change_res = change_service.detect_change(
            self.sample_tif_0,
            self.sample_tif,
            date_a="2024-01-15",
            date_b="2024-06-20"
        )
        self.assertIn("percentage_change", change_res)
        self.assertIn("change_heatmap_preview", change_res)
        self.assertGreater(change_res["changed_pixels"], 0)
        self.assertGreaterEqual(change_res["percentage_change"], 0.0)
        print(f"[TEST PASS] Change detection verified: {change_res['percentage_change']}% change.")

    # ============================================================
    # 6. MULTISPECTRAL INDICES (NDVI/NDWI)
    # ============================================================
    def test_06_multispectral_indices(self):
        # Create synthetic RGB image
        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        dummy_rgb[:, :, 1] = 200 # green heavy
        spec_res = multimodal_service.compute_spectral_indices(dummy_rgb)
        self.assertIn("ndvi_preview", spec_res)
        self.assertIn("ndwi_preview", spec_res)
        self.assertIn("mean_ndvi", spec_res)
        print(f"[TEST PASS] Multispectral indices verified: NDVI={spec_res['mean_ndvi']}.")

    # ============================================================
    # 7. STRUCTURED QUERY ROUTER & INTENTS
    # ============================================================
    def test_07_query_router_intents(self):
        self.assertEqual(query_router.classify_intent("How many cargo trucks?"), "COUNT")
        self.assertEqual(query_router.classify_intent("Show all vans with confidence > 50%"), "LIST")
        self.assertEqual(query_router.classify_intent("What is the flood risk?"), "FLOOD_ANALYSIS")
        self.assertEqual(query_router.classify_intent("Is there any traffic hotspot?"), "TRAFFIC_ANALYSIS")
        self.assertEqual(query_router.classify_intent("What changed between June and August?"), "CHANGE")
        self.assertEqual(query_router.classify_intent("Generate intelligence report"), "REPORT")
        self.assertEqual(query_router.classify_intent("What is Sentinel-1 useful for?"), "KNOWLEDGE")

        # Test deterministic routing execution
        dummy_dets = [
            {"id": 1, "class_name": "Cargo Truck", "confidence": 0.85, "obb": {"center_px": [100, 100], "width_px": 30, "height_px": 10}},
            {"id": 2, "class_name": "Cargo Truck", "confidence": 0.75, "obb": {"center_px": [120, 110], "width_px": 28, "height_px": 9}},
            {"id": 3, "class_name": "Van", "confidence": 0.60, "obb": {"center_px": [200, 200], "width_px": 15, "height_px": 8}},
        ]
        routed = query_router.execute_routed_query("How many cargo trucks?", detection_results={"detections": dummy_dets})
        self.assertEqual(routed["intent"], "COUNT")
        self.assertEqual(routed["evidence_count"], 2)
        self.assertIn("Exactly `2` object(s)", routed["structured_findings"])
        print("[TEST PASS] Query router & deterministic intent execution verified.")

    # ============================================================
    # 8. INTELLIGENCE REPORT GENERATION
    # ============================================================
    def test_08_intelligence_report_generation(self):
        rep = report_service.generate_report(
            title="Operational Test Mission",
            aoi_name="AOI Grid Alpha",
            detections=[{"id": 1, "class_name": "Cargo Truck", "confidence": 0.80}],
            sensor_info={"sensor": "Sentinel-2 & High-Res Optical", "resolution": "0.5m GSD"}
        )
        self.assertIn("report_id", rep)
        self.assertIn("markdown_content", rep)
        self.assertIn("html_content", rep)
        self.assertIn("Operational Test Mission", rep["markdown_content"])
        print(f"[TEST PASS] Intelligence report generator verified: ID {rep['report_id']}.")

    # ============================================================
    # 9. FASTAPI REST API ENDPOINTS
    # ============================================================
    def test_09_fastapi_endpoints(self):
        # 1. Health
        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["status"], "ok")

        # 2. Models
        r_models = self.client.get("/api/models")
        self.assertEqual(r_models.status_code, 200)
        self.assertIn("providers", r_models.json())

        # 3. Demo scenarios
        r_demo = self.client.get("/api/demo/scenarios")
        self.assertEqual(r_demo.status_code, 200)
        self.assertEqual(len(r_demo.json()["scenarios"]), 5)

        # 4. Traffic Intelligence Endpoint
        r_traffic = self.client.post("/api/intelligence/traffic", json={
            "detections": [
                {"id": 1, "class_name": "Small Car", "obb": {"center_px": [100, 100]}},
                {"id": 2, "class_name": "Small Car", "obb": {"center_px": [120, 110]}},
                {"id": 3, "class_name": "Van", "obb": {"center_px": [130, 105]}}
            ]
        })
        self.assertEqual(r_traffic.status_code, 200)
        self.assertEqual(r_traffic.json()["total_vehicles"], 3)
        print("[TEST PASS] All FastAPI REST endpoints verified.")


if __name__ == "__main__":
    unittest.main()
