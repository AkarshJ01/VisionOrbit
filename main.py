import os
import uuid
import json
import uvicorn
import asyncio
import base64
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Body
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from typing import Optional, List, Dict, Any

import rag

from backend.models import (
    ChatRequest,
    ChatResponse,
    ImageAnalysisRequest,
    RagRequest,
    QueryRouteRequest,
    AOIFilterRequest,
    AOIFilterResponse,
    SARProcessRequest,
    SARProcessResponse,
    SpectralAnalysisRequest,
    SpectralAnalysisResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    TrafficHotspotRequest,
    TrafficHotspotResponse,
    FloodRiskRequest,
    FloodRiskResponse,
    IntelligenceReportRequest,
    IntelligenceReportResponse
)

from backend.vision_service import vision_service
from backend.detection_service import detection_service
from backend.geospatial_service import geospatial_service
from backend.multimodal_service import multimodal_service
from backend.change_service import change_service
from backend.query_router import query_router
from backend.report_service import report_service

# ============================================================
# SAR SENTINEL CNN
# ============================================================

from SAR.src.inference.sentinel_cnn import (
    run_sentinel_inference,
    extract_detections
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SatQuery AI / VisionOrbit API",
    description="Multi-Modal Earth Observation AI Perception, GIS Reasoning & pinecone RAG",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sar")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAR_OUTPUT_DIR, exist_ok=True)


# ============================================================
# STATIC MOUNT
# ============================================================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SatQuery AI / VisionOrbit API is running."}


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "SatQuery AI Multi-Modal Earth Observation Platform",
        "has_pinecone": vision_service.has_pinecone,
        "index_name": vision_service.pinecone_index,
        "rag_module": "rag.py",
        "sar_model": "SAR/models/best_model.pth",
        "detection_model": "SAR/models/best.pt",
        "version": "2.0.0"
    }


# ============================================================
# MODELS & PROVIDERS
# ============================================================

@app.get("/api/models")
async def get_models(ollama_url: Optional[str] = None):
    ollama_info = await vision_service.check_ollama_status(ollama_url)
    has_env_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_env_tavily = bool(os.getenv("TAVILY_API_KEY", "").strip())
    has_pinecone = vision_service.has_pinecone

    ollama_models = []
    if ollama_info.get("available"):
        for m in ollama_info.get("all_models", []):
            if "embed" in m.lower() or "nomic" in m.lower():
                continue
            badge_desc = "Local LLM + Pinecone RAG"
            if any(v in m.lower() for v in ["llava", "vision", "moondream"]):
                badge_desc = "Local Multimodal Vision"
            ollama_models.append({
                "id": m,
                "name": f"Ollama {m}",
                "description": badge_desc
            })

    return {
        "providers": [
            {
                "id": "builtin",
                "name": "SatQuery AI Core Engine",
                "badge": "Active & Fast",
                "available": True,
                "models": [
                    {
                        "id": "visionorbit-core",
                        "name": "SatQuery Grounded Intelligence Engine",
                        "description": "Deterministic GIS + YOLO-OBB + Sentinel CNN"
                    }
                ]
            },
            {
                "id": "ollama",
                "name": "Ollama Local",
                "badge": "Connected" if ollama_info.get("available") else "Offline",
                "available": ollama_info.get("available", False),
                "url": ollama_info.get("url"),
                "models": ollama_models or [
                    {
                        "id": "gpt-oss:20b",
                        "name": "Ollama gpt-oss:20b",
                        "description": "RAG Reasoning Model"
                    }
                ]
            },
            {
                "id": "openai",
                "name": "OpenAI Vision",
                "badge": "Configured in .env" if has_env_openai else "API Key Required",
                "available": True,
                "configured": has_env_openai,
                "models": [
                    {
                        "id": "gpt-4o",
                        "name": "GPT-4o",
                        "description": "Multimodal vision model"
                    },
                    {
                        "id": "gpt-4o-mini",
                        "name": "GPT-4o Mini",
                        "description": "Fast multimodal model"
                    }
                ]
            }
        ],
        "features": {
            "has_tavily": has_env_tavily,
            "has_openai": has_env_openai,
            "has_ollama": ollama_info.get("available", False),
            "has_pinecone": has_pinecone,
            "pinecone_index": vision_service.pinecone_index,
            "sar_npy": True,
            "multispectral": True,
            "change_detection": True,
            "geospatial_reasoning": True
        }
    }


# ============================================================
# RAG DIRECT ENDPOINT
# ============================================================

@app.post("/api/rag")
async def rag_direct_endpoint(request: RagRequest):
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: rag.retrival_chain_with_sources(
                query=request.query,
                model_name=request.model or "gpt-oss:20b",
                k=request.k or 3
            )
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# OBJECT DETECTION DIRECT ENDPOINT
# ============================================================

@app.post("/api/detect")
async def detect_direct_endpoint(request: ImageAnalysisRequest):
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: detection_service.detect(
                request.image,
                conf_threshold=request.confThreshold or 0.20
            )
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# SAR SENTINEL CNN PIPELINE
# ============================================================

async def run_sar_pipeline(npy_path: str, prompt: str, model_name: Optional[str] = None):
    loop = asyncio.get_running_loop()

    sar_data = await loop.run_in_executor(
        None,
        lambda: multimodal_service.process_sar_npy(
            npy_path=npy_path,
            colormap_name="inferno",
            speckle_filter=True,
            threshold=0.5
        )
    )

    detector_output = {
        "model": "Sentinel CNN U-Net",
        "model_path": "SAR/models/best_model.pth",
        "input_channels": sar_data["channels"],
        "input_shape": sar_data["shape"],
        "detections": sar_data["regions"],
        "detected_regions": sar_data["detected_regions_count"],
        "positive_pixels": sar_data["positive_pixels"],
        "coverage_pct": sar_data["coverage_pct"],
        "mean_backscatter_db": sar_data["mean_backscatter_db"]
    }

    rag_result = await loop.run_in_executor(
        None,
        lambda: rag.detector_rag_chain(
            query=prompt,
            detector_output=detector_output,
            model_name=model_name or "gpt-oss:20b",
            k=3
        )
    )

    evidence_items = []
    for r in sar_data["regions"]:
        x, y, w, h = r["bbox_px"]
        cx, cy = r["center_px"]
        evidence_items.append({
            "id": f"sar_region_{r['id']}",
            "type": "mask_region",
            "label": f"SAR Detected Feature #{r['id']}",
            "confidence": 0.90,
            "confidence_percent": "90.0%",
            "center_px": [cx, cy],
            "bbox_px": [x, y, w, h],
            "area_px": r["area_px"],
            "provenance": "Sentinel CNN 7-Channel U-Net"
        })

    return {
        "reply": rag_result.get("reply", ""),
        "provider_used": rag_result.get("provider_used", "Sentinel CNN + RAG + LLM"),
        "model_used": rag_result.get("model_used", "gpt-oss:20b"),
        "annotated_image": sar_data["composite_image"],
        "preview_image": sar_data["preview_image"],
        "mask_image": sar_data["mask_image"],
        "detection_results": detector_output,
        "image_metadata": [{
            "format": "NumPy SAR Array",
            "width": sar_data["shape"][2],
            "height": sar_data["shape"][1],
            "channels": sar_data["channels"],
            "dtype": sar_data["dtype"]
        }],
        "search_sources": [],
        "rag_sources": rag_result.get("rag_sources", []),
        "evidence_items": evidence_items,
        "intent_detected": "SAR_ANALYSIS"
    }


# ============================================================
# CHAT ENDPOINT (GROUNDED MULTI-MODAL CONVERSATION)
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        images = request.images or []
        npy_path = None

        for image in images:
            if isinstance(image, str) and image.startswith("npy://"):
                npy_id = image[len("npy://"):]
                candidate = os.path.join(UPLOAD_DIR, npy_id)
                if os.path.isfile(candidate):
                    npy_path = candidate
                    break

        if npy_path is not None:
            result = await run_sar_pipeline(
                npy_path=npy_path,
                prompt=request.prompt,
                model_name=request.model
            )
            return ChatResponse(
                reply=result["reply"],
                provider_used=result["provider_used"],
                model_used=result["model_used"],
                annotated_image=result["annotated_image"],
                detection_results=result["detection_results"],
                image_metadata=result["image_metadata"],
                search_sources=result["search_sources"],
                rag_sources=result["rag_sources"],
                evidence_items=result.get("evidence_items", []),
                intent_detected=result.get("intent_detected", "SAR_ANALYSIS")
            )

        # Standard Image / Multi-Turn Pipeline
        result = await vision_service.generate_response(
            prompt=request.prompt,
            images=images,
            history=request.history or [],
            provider=request.provider or "auto",
            model=request.model or "gpt-oss:20b",
            api_key=request.apiKey,
            ollama_base_url=request.ollamaBaseUrl,
            tavily_key=request.tavilyApiKey,
            use_web_search=request.useWebSearch or False,
            use_rag=request.useRag if request.useRag is not None else True,
            system_prompt=request.systemPrompt,
            conf_threshold=request.confThreshold if request.confThreshold is not None else 0.20,
            use_detection=request.useDetection if request.useDetection is not None else True,
            aoi_bounds=request.aoi_bounds,
            aoi_polygon=request.aoi_polygon
        )

        return ChatResponse(
            reply=result.get("reply", ""),
            provider_used=result.get("provider_used", "SatQuery Engine"),
            model_used=result.get("model_used", "SatQuery Core"),
            annotated_image=result.get("annotated_image"),
            detection_results=result.get("detection_results"),
            image_metadata=result.get("image_metadata", []),
            search_sources=result.get("search_sources", []),
            rag_sources=result.get("rag_sources", []),
            evidence_items=result.get("evidence_items", []),
            intent_detected=result.get("intent_detected"),
            spatial_summary=result.get("spatial_summary")
        )

    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FILE UPLOAD ENDPOINT
# ============================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename or "uploaded_file"
        filename_lower = filename.lower()
        contents = await file.read()

        # NPY file
        if filename_lower.endswith(".npy"):
            file_id = f"{uuid.uuid4().hex}.npy"
            save_path = os.path.join(UPLOAD_DIR, file_id)
            with open(save_path, "wb") as f:
                f.write(contents)

            arr = np.load(save_path, allow_pickle=False)
            shape = list(arr.shape)
            dtype = str(arr.dtype)

            return {
                "filename": filename,
                "file_id": file_id,
                "file_type": "npy",
                "pipeline": "sar",
                "npy_reference": f"npy://{file_id}",
                "shape": shape,
                "dtype": dtype,
                "size_kb": round(len(contents) / 1024, 2),
                "message": "SAR .npy uploaded successfully."
            }

        # TIFF / GeoTIFF
        if filename_lower.endswith((".tif", ".tiff")) or file.content_type in ["image/tiff", "image/tif"]:
            img_rgb, fmt, w, h = detection_service.decode_image_data(contents)
            data_url = detection_service.encode_image_to_data_url(img_rgb)
            metadata = {
                "format": "GeoTIFF / TIFF",
                "width": w,
                "height": h,
                "aspect_ratio": f"{round(w / h, 2)}:1" if h > 0 else "1:1",
                "size_kb": round(len(contents) / 1024, 2),
                "is_valid": True
            }
            return {
                "filename": filename,
                "data_url": data_url,
                "mime_type": "image/jpeg",
                "metadata": metadata,
                "file_type": "image",
                "pipeline": "optical"
            }

        # Standard PNG / JPG / WEBP
        mime_type = file.content_type or "image/png"
        b64 = base64.b64encode(contents).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        metadata = vision_service.parse_image_info(data_url)

        return {
            "filename": filename,
            "data_url": data_url,
            "mime_type": mime_type,
            "metadata": metadata,
            "file_type": "image",
            "pipeline": "optical"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process upload: {str(e)}")


# ============================================================
# NEW ADVANCED ENDPOINTS: GEOSPATIAL, SAR, CHANGE, REPORT
# ============================================================

@app.post("/api/query/route")
async def route_query_endpoint(request: QueryRouteRequest):
    """Executes structured intent routing and deterministic analysis."""
    try:
        res = query_router.execute_routed_query(
            query=request.query,
            detection_results={"detections": request.detections} if request.detections else None,
            aoi_polygon=request.aoi_polygon,
            aoi_bounds=request.aoi_bounds
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/geospatial/aoi", response_model=AOIFilterResponse)
async def aoi_filter_endpoint(request: AOIFilterRequest):
    """Filters detected targets inside user-drawn AOI Polygon."""
    try:
        coords = request.aoi.coordinates
        is_latlon = request.is_latlon or any(d.get("latitude") is not None for d in request.detections)
        res = geospatial_service.filter_detections_in_aoi(
            detections=request.detections,
            aoi_coords=coords,
            is_latlon=is_latlon
        )
        return AOIFilterResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sar/process", response_model=SARProcessResponse)
async def sar_process_endpoint(request: SARProcessRequest):
    """Scientific SAR .npy processing, log-dB scaling, speckle filtering, and segmentation."""
    try:
        npy_path = os.path.join(UPLOAD_DIR, request.npy_file_id)
        if not os.path.exists(npy_path):
            raise HTTPException(status_code=404, detail="NPY file not found.")

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: multimodal_service.process_sar_npy(
                npy_path=npy_path,
                colormap_name=request.colormap or "inferno",
                speckle_filter=request.speckle_filter if request.speckle_filter is not None else True,
                threshold=request.threshold or 0.50
            )
        )
        return SARProcessResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multispectral/indices", response_model=SpectralAnalysisResponse)
async def multispectral_indices_endpoint(request: SpectralAnalysisRequest):
    """Computes vegetation (NDVI) and water (NDWI) indices."""
    try:
        img_rgb, _, _, _ = detection_service.decode_image_data(request.image_data)
        res = multimodal_service.compute_spectral_indices(img_rgb)
        return SpectralAnalysisResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/change/detect", response_model=ChangeDetectionResponse)
async def change_detection_endpoint(request: ChangeDetectionRequest):
    """Multi-temporal change detection between Date A and Date B images."""
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: change_service.detect_change(
                image_a_input=request.image_a,
                image_b_input=request.image_b,
                date_a=request.date_a or "Date A",
                date_b=request.date_b or "Date B",
                sensitivity=request.sensitivity or 0.30,
                min_change_area_px=request.min_change_area_px or 50
            )
        )
        return ChangeDetectionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/traffic", response_model=TrafficHotspotResponse)
async def traffic_hotspot_endpoint(request: TrafficHotspotRequest):
    """Identifies vehicle density aggregations and potential traffic hotspots."""
    try:
        res = geospatial_service.analyze_traffic_hotspots(
            detections=request.detections,
            image_w=request.image_width or 1000,
            image_h=request.image_height or 1000,
            distance_threshold_px=request.distance_threshold_px or 80.0
        )
        return TrafficHotspotResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/flood", response_model=FloodRiskResponse)
async def flood_risk_endpoint(request: FloodRiskRequest):
    """Calculates multi-sensor flood risk indicator."""
    try:
        res = multimodal_service.assess_flood_risk(
            water_mask_coverage_pct=request.water_mask_coverage_pct,
            elevation_slope_deg=request.elevation_slope_deg,
            rainfall_indicator_mm=request.rainfall_indicator_mm
        )
        return FloodRiskResponse(
            risk_level=res["risk_level"],
            confidence=res["confidence"],
            water_extent_pct=res["water_extent_pct"],
            contributing_factors=res["contributing_factors"],
            terrain_vulnerability=res["terrain_vulnerability"],
            evidence_summary=res["evidence_summary"],
            limitations=res["limitations"],
            evidence_items=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report/generate", response_model=IntelligenceReportResponse)
async def generate_report_endpoint(request: IntelligenceReportRequest):
    """Generates structured Earth Observation intelligence report."""
    try:
        res = report_service.generate_report(
            title=request.title or "VisionOrbit Earth Observation Intelligence Report",
            aoi_name=request.aoi_name or "Target AOI Footprint",
            detections=request.detections or [],
            sar_telemetry=request.sar_telemetry,
            change_telemetry=request.change_telemetry,
            traffic_telemetry=request.traffic_telemetry,
            flood_telemetry=request.flood_telemetry,
            sensor_info=request.sensor_info
        )
        return IntelligenceReportResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DEMO SCENARIOS
# ============================================================

@app.get("/api/demo/scenarios")
async def get_demo_scenarios():
    """Returns curated verified demo scenarios for rapid SIH demonstration."""
    return {
        "scenarios": [
            {
                "id": "urban_detection",
                "name": "Urban Aerial Target Intelligence",
                "tag": "37-Class YOLO-OBB",
                "description": "Sub-meter GeoTIFF aerial scene with 61 vehicles, vans, and cargo trucks.",
                "sample_image": "data/raw/images/1.tif",
                "default_prompt": "Detect and identify all vehicles, cargo trucks, and spatial coordinates in this AOI.",
                "type": "optical"
            },
            {
                "id": "sar_radar",
                "name": "Sentinel-1 SAR Radar Analysis",
                "tag": "SAR 7-Channel CNN",
                "description": "7-channel Sentinel SAR backscatter array with U-Net radar segmentation.",
                "sample_npy": "uploads/bfb19c2a6dbd4c83b95007fcd6b01502.npy",
                "default_prompt": "Analyze backscatter amplitude, Lee speckle filtering, and segment radar features.",
                "type": "sar"
            },
            {
                "id": "temporal_change",
                "name": "Temporal Environmental Change",
                "tag": "Multi-Temporal Diff",
                "description": "Bitemporal change detection measuring structural and radiometric shifts.",
                "sample_image_a": "data/raw/images/0.tif",
                "sample_image_b": "data/raw/images/1.tif",
                "default_prompt": "What changed between Date A and Date B?",
                "type": "change"
            },
            {
                "id": "flood_risk",
                "name": "Flood & Inundation Risk Assessment",
                "tag": "Decision Support",
                "description": "Multi-sensor flood vulnerability combining radar moisture and terrain elevation.",
                "default_prompt": "Assess flood risk indicator based on surface moisture extent and terrain slope.",
                "type": "flood"
            },
            {
                "id": "traffic_hotspot",
                "name": "Traffic Density Hotspot Assessment",
                "tag": "Spatial GIS",
                "description": "Graph clustering on vehicle coordinates to infer localized traffic hotspots.",
                "sample_image": "data/raw/images/1.tif",
                "default_prompt": "Are there potential vehicle traffic hotspots in this area?",
                "type": "traffic"
            }
        ]
    }


# ============================================================
# MAIN
# ============================================================

def main():
    port = int(os.getenv("PORT", 8000))
    print(
        f"""
============================================================
🛰️ SatQuery AI / VisionOrbit — Multi-Modal Earth Intelligence
============================================================
Server: http://localhost:{port}

Active Pipelines:
  1. 37-Class YOLO-OBB Aerial Detection (Oriented Bounding Boxes)
  2. Sentinel CNN 7-Channel SAR Segmentation (.npy Scientific Pipeline)
  3. Deterministic Geospatial AOI & Distance Engine (GeoPandas / Shapely)
  4. Temporal Change Detection & Radiometric Difference Engine
  5. Multispectral Indices (NDVI / NDWI) & Decision Support Indicators
  6. Curated Remote Sensing RAG & Knowledge Engine
  7. Automated Executive Intelligence Report Generator
============================================================
"""
    )
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()