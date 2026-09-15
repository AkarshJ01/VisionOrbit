import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
import pyproj

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False


class GeospatialService:
    """
    Deterministic Geospatial Intelligence Service.
    Executes geometric filtering, point-in-polygon intersection, proximity,
    clustering, density, and spatial telemetry.
    """

    def __init__(self):
        # WGS84 Geodetic
        self.geod = pyproj.Geod(ellps="WGS84")

    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculates 2D surface distance in meters between two lat/lon pairs on Earth."""
        if HAS_GEOPY:
            try:
                return float(geodesic((lat1, lon1), (lat2, lon2)).meters)
            except Exception:
                pass
        
        # Great-circle Haversine formula fallback
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(R * c)

    def calculate_polygon_area_m2(self, coordinates: List[List[float]]) -> float:
        """
        Calculates geodesic area in square meters for a WGS84 polygon [[lat, lon], ...].
        """
        if len(coordinates) < 3:
            return 0.0
        try:
            # Note: Geod.polygon_area_perimeter expects lons, lats
            lons = [pt[1] for pt in coordinates]
            lats = [pt[0] for pt in coordinates]
            area, _ = self.geod.polygon_area_perimeter(lons, lats)
            return abs(float(area))
        except Exception:
            # Approximate planar area if geodesic calculation fails
            poly = Polygon(coordinates)
            return float(poly.area * 111320 * 111320)

    def filter_detections_in_aoi(
        self,
        detections: List[Dict[str, Any]],
        aoi_coords: List[Any],
        is_latlon: bool = True
    ) -> Dict[str, Any]:
        """
        Deterministically filters detections that fall inside a user-drawn AOI Polygon or Bounding Box.
        Supports both geographic (Lat/Lon) and Pixel coordinate spaces.
        """
        if not detections:
            return {
                "total_in_aoi": 0,
                "total_outside_aoi": 0,
                "matching_detections": [],
                "class_counts_in_aoi": {},
                "density_per_sq_km": 0.0,
                "aoi_area_m2": 0.0,
                "evidence_ids": []
            }

        # Build Shapely Polygon
        try:
            if len(aoi_coords) == 4 and isinstance(aoi_coords[0], (int, float)):
                # Bounding box [min_x/lat, min_y/lon, max_x/lat, max_y/lon]
                poly = box(aoi_coords[0], aoi_coords[1], aoi_coords[2], aoi_coords[3])
            else:
                # Polygon ring [[x, y], ...]
                poly = Polygon(aoi_coords)
        except Exception as e:
            print(f"[GeospatialService] Invalid AOI polygon: {e}")
            return {
                "total_in_aoi": len(detections),
                "total_outside_aoi": 0,
                "matching_detections": detections,
                "class_counts_in_aoi": {},
                "density_per_sq_km": 0.0,
                "aoi_area_m2": 0.0,
                "evidence_ids": [d.get("id") for d in detections if "id" in d]
            }

        inside_dets = []
        outside_count = 0
        evidence_ids = []
        class_counts: Dict[str, int] = {}

        for d in detections:
            # Determine point coordinate (Lat/Lon or Pixel center)
            if is_latlon and d.get("latitude") is not None and d.get("longitude") is not None:
                pt = Point(d["latitude"], d["longitude"])
            else:
                # Fallback to pixel space
                cx, cy = d.get("obb", {}).get("center_px", [0, 0])
                pt = Point(cx, cy)

            if poly.contains(pt) or poly.touches(pt):
                inside_dets.append(d)
                det_id = d.get("id")
                if det_id is not None:
                    evidence_ids.append(det_id)
                cls_name = d.get("class_name", "Unknown")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            else:
                outside_count += 1

        # Calculate area and density
        area_m2 = 0.0
        density_sq_km = 0.0
        if is_latlon and isinstance(aoi_coords[0], list):
            area_m2 = self.calculate_polygon_area_m2(aoi_coords)
            if area_m2 > 0:
                area_sq_km = area_m2 / 1_000_000.0
                density_sq_km = round(len(inside_dets) / area_sq_km, 2)

        return {
            "total_in_aoi": len(inside_dets),
            "total_outside_aoi": outside_count,
            "matching_detections": inside_dets,
            "class_counts_in_aoi": class_counts,
            "density_per_sq_km": density_sq_km,
            "aoi_area_m2": round(area_m2, 2),
            "evidence_ids": evidence_ids
        }

    def compute_spatial_clusters(
        self,
        detections: List[Dict[str, Any]],
        distance_threshold_px: float = 90.0,
        min_cluster_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Performs graph-based density clustering on detected objects in pixel space.
        Identifies spatial aggregations and candidate hotspots.
        """
        if not detections:
            return []

        pts = []
        det_indices = []
        for idx, d in enumerate(detections):
            cx, cy = d.get("obb", {}).get("center_px", [0, 0])
            pts.append((cx, cy))
            det_indices.append(idx)

        n = len(pts)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            
            # BFS to find connected component within distance threshold
            cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr = queue.pop(0)
                cx1, cy1 = pts[curr]

                for j in range(n):
                    if not visited[j]:
                        cx2, cy2 = pts[j]
                        dist = math.hypot(cx1 - cx2, cy1 - cy2)
                        if dist <= distance_threshold_px:
                            visited[j] = True
                            cluster.append(j)
                            queue.append(j)

            if len(cluster) >= min_cluster_size:
                cluster_dets = [detections[k] for k in cluster]
                cluster_pts = [pts[k] for k in cluster]
                mean_x = float(np.mean([p[0] for p in cluster_pts]))
                mean_y = float(np.mean([p[1] for p in cluster_pts]))
                
                # Bounding box of cluster
                min_x = min(p[0] for p in cluster_pts)
                max_x = max(p[0] for p in cluster_pts)
                min_y = min(p[1] for p in cluster_pts)
                max_y = max(p[1] for p in cluster_pts)

                clusters.append({
                    "cluster_id": len(clusters) + 1,
                    "size": len(cluster),
                    "center_px": [round(mean_x, 1), round(mean_y, 1)],
                    "bbox_px": [round(min_x, 1), round(min_y, 1), round(max_x - min_x, 1), round(max_y - min_y, 1)],
                    "detection_ids": [d.get("id") for d in cluster_dets if "id" in d],
                    "classes": list(set(d.get("class_name") for d in cluster_dets if "class_name" in d))
                })

        # Sort clusters by size descending
        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters

    def analyze_traffic_hotspots(
        self,
        detections: List[Dict[str, Any]],
        image_w: int = 1000,
        image_h: int = 1000,
        distance_threshold_px: float = 85.0
    ) -> Dict[str, Any]:
        """
        Decision-support traffic intelligence module.
        Analyzes vehicle density and spatial aggregation across aerial footprints.
        Outputs scientifically defensible potential hotspot indicator.
        """
        vehicle_classes = {
            "small car", "van", "dump truck", "cargo truck", "truck tractor",
            "bus", "trailer", "other-vehicle", "small-vehicle", "large-vehicle"
        }

        vehicles = [
            d for d in detections 
            if d.get("class_name", "").lower() in vehicle_classes
        ]

        total_vehicles = len(vehicles)
        total_area_mp = (image_w * image_h) / 1_000_000.0  # Megapixels
        density_per_mp = round(total_vehicles / total_area_mp, 1) if total_area_mp > 0 else 0

        clusters = self.compute_spatial_clusters(
            vehicles,
            distance_threshold_px=distance_threshold_px,
            min_cluster_size=3
        )

        hotspot_evidence_ids = []
        for c in clusters:
            hotspot_evidence_ids.extend(c.get("detection_ids", []))

        # Determine risk indicator level
        if len(clusters) >= 2 or (clusters and clusters[0]["size"] >= 8) or total_vehicles > 25:
            risk_level = "Elevated Potential Hotspot"
            confidence = "Moderate Confidence"
        elif len(clusters) == 1 or total_vehicles >= 10:
            risk_level = "Localized Vehicle Density"
            confidence = "Moderate Confidence"
        else:
            risk_level = "Nominal Flow / Low Density"
            confidence = "High Confidence"

        summary_md = (
            f"**Traffic Intelligence Telemetry:**\n"
            f"- **Identified Vehicles:** `{total_vehicles}` targets across `{image_w}×{image_h}` footprint.\n"
            f"- **Vehicle Density:** `{density_per_mp}` vehicles / Megapixel.\n"
            f"- **Identified Spatial Clusters:** `{len(clusters)}` localized aggregation clusters.\n"
            f"- **Assessment:** `{risk_level}` ({confidence}).\n\n"
            f"> ℹ️ *Note: Telemetry represents instantaneous spatial density indicator from aerial/satellite imagery, not continuous velocity tracking.*"
        )

        return {
            "risk_level": risk_level,
            "total_vehicles": total_vehicles,
            "vehicle_density_ratio": density_per_mp,
            "hotspot_clusters_count": len(clusters),
            "hotspot_clusters": clusters,
            "confidence": confidence,
            "summary_markdown": summary_md,
            "evidence_ids": hotspot_evidence_ids
        }


geospatial_service = GeospatialService()
