from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field
import time
from enum import Enum


# ============================================================
# BASIC CHAT & MESSAGE MODELS (PRESERVED)
# ============================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Textual message content")
    images: Optional[List[str]] = Field(default=[], description="List of base64 data URLs or image URLs")
    detection_results: Optional[Dict[str, Any]] = Field(default=None, description="Prior object detection findings")
    annotated_image: Optional[str] = Field(default=None, description="Prior annotated image")
    evidence_items: Optional[List[Dict[str, Any]]] = Field(default=[], description="Visual evidence linked to message")
    timestamp: Optional[float] = Field(default_factory=time.time)


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or question")
    images: Optional[List[str]] = Field(default=[], description="List of base64 images or image URLs")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation context")
    provider: Optional[str] = Field(default="auto", description="'openai', 'ollama', 'builtin', or 'auto'")
    model: Optional[str] = Field(default="gpt-oss:20b", description="Target model name")
    apiKey: Optional[str] = Field(default=None, description="Client-provided OpenAI API Key (optional)")
    ollamaBaseUrl: Optional[str] = Field(default=None, description="Client-provided Ollama Base URL (optional)")
    tavilyApiKey: Optional[str] = Field(default=None, description="Client-provided Tavily API Key (optional)")
    useWebSearch: Optional[bool] = Field(default=False, description="Whether to enrich with Tavily web search")
    useRag: Optional[bool] = Field(default=True, description="Whether to retrieve context from Pinecone Knowledge Base")
    systemPrompt: Optional[str] = Field(default=None, description="Custom system instruction")
    confThreshold: Optional[float] = Field(default=0.20, description="YOLO object detection confidence threshold")
    useDetection: Optional[bool] = Field(default=True, description="Whether to execute real-time YOLO-OBB object detection")
    aoi_bounds: Optional[List[float]] = Field(default=None, description="Optional bounding box [min_lat, min_lon, max_lat, max_lon] or [x1, y1, x2, y2]")
    aoi_polygon: Optional[List[List[float]]] = Field(default=None, description="Optional polygon coordinates [[lat, lon], ...]")


class RagRequest(BaseModel):
    query: str = Field(..., description="Search query or question for Pinecone RAG")
    model: Optional[str] = Field(default="gpt-oss:20b", description="Target Ollama model name")
    k: Optional[int] = Field(default=3, description="Number of document chunks to retrieve")


class ImageAnalysisRequest(BaseModel):
    image: str = Field(..., description="Base64 image data URL")
    aspects: Optional[List[str]] = Field(default=["description", "objects", "text", "colors", "recommendations"])
    provider: Optional[str] = Field(default="auto")
    apiKey: Optional[str] = None
    confThreshold: Optional[float] = 0.20


# ============================================================
# STRUCTURED EVIDENCE & GEOSPATIAL SCHEMAS
# ============================================================

class EvidenceItem(BaseModel):
    id: Union[int, str]
    type: str = Field(..., description="'detection', 'polygon', 'mask_region', 'change_cluster', 'hotspot'")
    label: str
    confidence: Optional[float] = None
    confidence_percent: Optional[str] = None
    center_px: Optional[List[float]] = None
    bbox_px: Optional[List[float]] = None
    polygon_px: Optional[List[List[float]]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    polygon_latlon: Optional[List[List[float]]] = None
    area_px: Optional[float] = None
    area_m2: Optional[float] = None
    provenance: Optional[str] = "YOLO-OBB / Sentinel CNN"
    metadata: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    reply: str
    provider_used: str
    model_used: str
    annotated_image: Optional[str] = None
    detection_results: Optional[Dict[str, Any]] = None
    image_metadata: Optional[List[Dict[str, Any]]] = []
    search_sources: Optional[List[Dict[str, Any]]] = []
    rag_sources: Optional[List[Dict[str, Any]]] = []
    evidence_items: Optional[List[EvidenceItem]] = []
    intent_detected: Optional[str] = None
    spatial_summary: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)


# ============================================================
# QUERY ROUTER SCHEMAS
# ============================================================

class QueryIntent(str, Enum):
    COUNT = "COUNT"
    LIST = "LIST"
    FILTER = "FILTER"
    SPATIAL = "SPATIAL"
    STATISTICS = "STATISTICS"
    CHANGE = "CHANGE"
    SEGMENTATION = "SEGMENTATION"
    FLOOD_ANALYSIS = "FLOOD_ANALYSIS"
    TRAFFIC_ANALYSIS = "TRAFFIC_ANALYSIS"
    KNOWLEDGE = "KNOWLEDGE"
    REPORT = "REPORT"
    GENERAL = "GENERAL"


class QueryRouteRequest(BaseModel):
    query: str
    image_data: Optional[str] = None
    npy_file_id: Optional[str] = None
    detections: Optional[List[Dict[str, Any]]] = None
    aoi_polygon: Optional[List[List[float]]] = None
    aoi_bounds: Optional[List[float]] = None
    history: Optional[List[Dict[str, Any]]] = []
    model: Optional[str] = "gpt-oss:20b"
    provider: Optional[str] = "auto"
    conf_threshold: Optional[float] = 0.20


# ============================================================
# GEOSPATIAL & AOI SCHEMAS
# ============================================================

class AOIGeometry(BaseModel):
    type: str = Field(default="Polygon", description="'Polygon', 'Rectangle', or 'Point'")
    coordinates: List[Any] = Field(..., description="Coordinates list [[lat, lon], ...] or [lat, lon]")
    name: Optional[str] = "User AOI"


class AOIFilterRequest(BaseModel):
    detections: List[Dict[str, Any]]
    aoi: AOIGeometry
    image_width: Optional[int] = 1000
    image_height: Optional[int] = 1000
    is_latlon: Optional[bool] = False


class AOIFilterResponse(BaseModel):
    total_in_aoi: int
    total_outside_aoi: int
    matching_detections: List[Dict[str, Any]]
    class_counts_in_aoi: Dict[str, int]
    density_per_sq_km: Optional[float] = None
    aoi_area_m2: Optional[float] = None
    evidence_ids: List[Union[int, str]] = []


# ============================================================
# MULTI-MODAL & SAR SCHEMAS
# ============================================================

class SARProcessRequest(BaseModel):
    npy_file_id: str
    colormap: Optional[str] = "inferno"  # 'inferno', 'viridis', 'gray', 'plasma'
    speckle_filter: Optional[bool] = True
    threshold: Optional[float] = 0.50
    extract_regions: Optional[bool] = True


class SARProcessResponse(BaseModel):
    preview_image: str  # Base64 data URL
    mask_image: Optional[str] = None  # Base64 data URL
    composite_image: Optional[str] = None  # Base64 data URL with mask overlay
    shape: List[int]
    dtype: str
    channels: int
    mean_backscatter_db: float
    min_val: float
    max_val: float
    positive_pixels: int
    detected_regions_count: int
    regions: List[Dict[str, Any]]
    provenance: Dict[str, Any]


class SpectralAnalysisRequest(BaseModel):
    image_data: str  # Base64 image
    compute_ndvi: Optional[bool] = True
    compute_ndwi: Optional[bool] = True
    compute_mndwi: Optional[bool] = False


class SpectralAnalysisResponse(BaseModel):
    rgb_preview: str
    ndvi_preview: Optional[str] = None
    ndwi_preview: Optional[str] = None
    mean_ndvi: Optional[float] = None
    mean_ndwi: Optional[float] = None
    vegetation_coverage_pct: Optional[float] = None
    water_coverage_pct: Optional[float] = None
    spectral_insights: List[str] = []


# ============================================================
# TEMPORAL CHANGE DETECTION SCHEMAS
# ============================================================

class ChangeDetectionRequest(BaseModel):
    image_a: str  # Base64 data URL or file path (Date A / Baseline)
    image_b: str  # Base64 data URL or file path (Date B / Follow-up)
    date_a: Optional[str] = "Date A"
    date_b: Optional[str] = "Date B"
    sensitivity: Optional[float] = 0.30
    min_change_area_px: Optional[int] = 50


class ChangeDetectionResponse(BaseModel):
    before_preview: str
    after_preview: str
    change_mask_preview: str
    change_heatmap_preview: str
    changed_pixels: int
    total_pixels: int
    percentage_change: float
    changed_regions_count: int
    changed_clusters: List[Dict[str, Any]]
    insights: List[str]
    evidence_items: List[EvidenceItem] = []


# ============================================================
# DECISION SUPPORT (TRAFFIC & FLOOD) SCHEMAS
# ============================================================

class TrafficHotspotRequest(BaseModel):
    detections: List[Dict[str, Any]]
    image_width: Optional[int] = 1000
    image_height: Optional[int] = 1000
    distance_threshold_px: Optional[float] = 80.0
    min_cluster_size: Optional[int] = 3


class TrafficHotspotResponse(BaseModel):
    risk_level: str  # 'Low', 'Moderate', 'Elevated Hotspot'
    total_vehicles: int
    vehicle_density_ratio: float
    hotspot_clusters_count: int
    hotspot_clusters: List[Dict[str, Any]]
    confidence: str
    summary_markdown: str
    evidence_ids: List[int] = []


class FloodRiskRequest(BaseModel):
    npy_file_id: Optional[str] = None
    image_data: Optional[str] = None
    water_mask_coverage_pct: Optional[float] = None
    elevation_slope_deg: Optional[float] = None
    rainfall_indicator_mm: Optional[float] = None


class FloodRiskResponse(BaseModel):
    risk_level: str  # 'Low', 'Moderate Indicator', 'Elevated Flood-Risk Indicator'
    confidence: str
    water_extent_pct: float
    contributing_factors: List[str]
    terrain_vulnerability: str
    evidence_summary: str
    limitations: str
    evidence_items: List[EvidenceItem] = []


# ============================================================
# INTELLIGENCE REPORT SCHEMAS
# ============================================================

class IntelligenceReportRequest(BaseModel):
    title: Optional[str] = "VisionOrbit Earth Observation Intelligence Report"
    aoi_name: Optional[str] = "Target Area of Interest"
    sensor_info: Optional[Dict[str, Any]] = None
    detections: Optional[List[Dict[str, Any]]] = []
    sar_telemetry: Optional[Dict[str, Any]] = None
    change_telemetry: Optional[Dict[str, Any]] = None
    traffic_telemetry: Optional[Dict[str, Any]] = None
    flood_telemetry: Optional[Dict[str, Any]] = None
    executive_summary: Optional[str] = None


class IntelligenceReportResponse(BaseModel):
    report_id: str
    created_at: str
    title: str
    markdown_content: str
    html_content: str
    summary: Dict[str, Any]
    provenance: Dict[str, Any]
