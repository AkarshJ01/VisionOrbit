import os
import math
import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import xy
    import pyproj
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def calculate_geo_distance(lat1, lon1, lat2, lon2):
    """
    Calculates geographic surface distance in meters between two lat/lon pairs on Earth.
    NOTE: This is 2D surface/geodesic distance on Earth, NOT sensor-to-object altitude.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    if HAS_GEOPY:
        try:
            return float(geodesic((lat1, lon1), (lat2, lon2)).meters)
        except Exception:
            pass
    return float(haversine_distance(lat1, lon1, lat2, lon2))

def format_distance(dist_meters):
    if dist_meters is None:
        return "N/A"
    if dist_meters < 1000:
        return f"{dist_meters:.1f} m"
    else:
        return f"{dist_meters/1000.0:.3f} km"

class GeoReferenceEngine:
    def __init__(self, image_path):
        self.image_path = os.path.expanduser(image_path)
        self.is_georeferenced = False
        self.crs_str = None
        self.transform = None
        self.transformer = None
        self.bounds = None
        
        self._inspect_geotiff()
        
    def _inspect_geotiff(self):
        if not HAS_RASTERIO:
            return
            
        if not self.image_path.lower().endswith(('.tif', '.tiff')):
            return
            
        try:
            with rasterio.open(self.image_path) as src:
                if src.crs is not None and src.transform is not None:
                    self.is_georeferenced = True
                    self.crs_str = str(src.crs)
                    self.transform = src.transform
                    
                    # Create projection transformer if not already WGS84
                    if src.crs != CRS.from_epsg(4326):
                        self.transformer = pyproj.Transformer.from_crs(
                            src.crs, "EPSG:4326", always_xy=True
                        )
                    else:
                        self.transformer = None
                        
                    b = src.bounds
                    self.bounds = {
                        "left": b.left,
                        "bottom": b.bottom,
                        "right": b.right,
                        "top": b.top
                    }
        except Exception as e:
            # File may be a non-georeferenced TIFF
            self.is_georeferenced = False
            
    def pixel_to_latlon(self, px_x, px_y):
        """
        Convert pixel coordinate (px_x, px_y) to (latitude, longitude) in WGS84.
        Returns: (lat, lon) as floats, or (None, None) if not georeferenced.
        """
        if not self.is_georeferenced or self.transform is None:
            return None, None
            
        try:
            # Transform pixel to map coordinate in source CRS
            # Note: rasterio transform * (col, row) or xy(transform, row, col)
            map_x, map_y = rasterio.transform.xy(self.transform, px_y, px_x, offset='center')
            
            if self.transformer is not None:
                lon, lat = self.transformer.transform(map_x, map_y)
            else:
                # Source CRS is EPSG:4326 (lon, lat)
                lon, lat = map_x, map_y
                
            return round(float(lat), 6), round(float(lon), 6)
        except Exception:
            return None, None
            
    def polygon_pixels_to_latlon(self, poly_pts):
        """Convert an array of pixel points [[x1, y1], [x2, y2], ...] to [[lat1, lon1], ...]"""
        if not self.is_georeferenced:
            return None
        geo_poly = []
        for pt in poly_pts:
            lat, lon = self.pixel_to_latlon(float(pt[0]), float(pt[1]))
            if lat is not None and lon is not None:
                geo_poly.append([lat, lon])
        return geo_poly if len(geo_poly) == len(poly_pts) else None
