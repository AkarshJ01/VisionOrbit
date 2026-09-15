import re
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.models import QueryIntent, EvidenceItem
from backend.geospatial_service import geospatial_service
from backend.multimodal_service import multimodal_service
from backend.change_service import change_service
import rag


class QueryRouter:
    """
    Structured Intent Router & Grounded Execution Engine for SatQuery AI.
    Routes natural language queries to deterministic analytical pipelines (Counts,
    GIS filters, Hotspots, Change, RAG) and produces evidence-grounded facts for LLM synthesis.
    """

    KNOWN_CLASSES = [
        "small car", "van", "dump truck", "cargo truck", "dry cargo ship",
        "motorboat", "intersection", "other-vehicle", "fishing boat", "other-ship",
        "other-airplane", "a220", "liquid cargo ship", "tennis court", "boeing737",
        "tugboat", "bus", "passenger ship", "engineering ship", "a321", "excavator",
        "trailer", "truck tractor", "baseball field", "football field", "warship",
        "basketball court", "tractor", "a330", "boeing787", "bridge", "boeing747",
        "boeing777", "arj21", "a350", "roundabout", "c919"
    ]

    def classify_intent(self, query: str, has_image: bool = False, has_npy: bool = False) -> QueryIntent:
        """Classifies query into structured analytical intent based on linguistic patterns."""
        q = query.lower().strip()

        # 1. Report Generation
        if any(k in q for k in ["generate report", "intelligence report", "mission report", "export report", "create report", "summary report"]):
            return QueryIntent.REPORT

        # 2. Traffic Analysis
        if any(k in q for k in ["traffic", "congestion", "hotspot", "vehicle density", "traffic jam", "bottleneck"]):
            return QueryIntent.TRAFFIC_ANALYSIS

        # 3. Flood Analysis
        if any(k in q for k in ["flood", "water risk", "inundation", "overflow", "submerged", "moisture risk"]):
            return QueryIntent.FLOOD_ANALYSIS

        # 4. Change Detection
        if any(k in q for k in ["change", "changed", "what changed", "difference", "compare dates", "before and after", "temporal"]):
            return QueryIntent.CHANGE

        # 5. Spatial / AOI / Proximity
        if any(k in q for k in ["inside aoi", "in polygon", "in area", "near", "closest", "proximity", "distance to", "coordinates of", "where is", "where are", "spatial distribution", "density"]):
            return QueryIntent.SPATIAL

        # 6. Count
        if any(k in q for k in ["how many", "count", "total number of", "number of", "how much"]):
            return QueryIntent.COUNT

        # 7. List / Filter
        if any(k in q for k in ["show all", "list all", "filter", "find all", "which detections", "identify all", "highest confidence"]):
            return QueryIntent.LIST

        # 8. Statistics
        if any(k in q for k in ["statistics", "breakdown", "average confidence", "distribution", "telemetry summary"]):
            return QueryIntent.STATISTICS

        # 9. Segmentation
        if any(k in q for k in ["segment", "segmentation", "mask", "extract regions", "connected components"]):
            return QueryIntent.SEGMENTATION

        # 10. Knowledge / RAG (Remote sensing domain queries)
        if any(k in q for k in ["what is sar", "what is sentinel", "band", "multispectral", "polarization", "stac", "gsd", "resolution of", "explain", "copernicus"]):
            return QueryIntent.KNOWLEDGE

        # Default fallback
        if has_image or has_npy:
            return QueryIntent.LIST
        return QueryIntent.KNOWLEDGE

    def extract_target_classes(self, query: str) -> List[str]:
        """Identifies specific FAIR1M / remote sensing target classes from the user query."""
        q = query.lower()
        matched = []

        # Generic vehicle / aircraft / ship synonyms mapping
        synonyms = {
            "plane": ["other-airplane", "a220", "boeing737", "a321", "a330", "boeing787", "boeing747", "boeing777", "arj21", "a350", "c919"],
            "airplane": ["other-airplane", "a220", "boeing737", "a321", "a330", "boeing787", "boeing747", "boeing777", "arj21", "a350", "c919"],
            "aircraft": ["other-airplane", "a220", "boeing737", "a321", "a330", "boeing787", "boeing747", "boeing777", "arj21", "a350", "c919"],
            "vehicle": ["small car", "van", "dump truck", "cargo truck", "other-vehicle", "bus", "trailer", "truck tractor", "tractor", "excavator"],
            "car": ["small car"],
            "truck": ["dump truck", "cargo truck", "truck tractor"],
            "ship": ["dry cargo ship", "motorboat", "fishing boat", "other-ship", "liquid cargo ship", "tugboat", "passenger ship", "engineering ship", "warship"],
            "boat": ["motorboat", "fishing boat", "tugboat"],
            "court": ["tennis court", "basketball court"],
            "field": ["baseball field", "football field"]
        }

        # Check exact known classes
        for cls in self.KNOWN_CLASSES:
            if cls in q or cls.replace(" ", "") in q or cls.replace("-", " ") in q:
                matched.append(cls)

        # Check synonyms if no exact match
        if not matched:
            for word, syn_list in synonyms.items():
                if word in q:
                    matched.extend(syn_list)

        return list(dict.fromkeys(matched))

    def execute_routed_query(
        self,
        query: str,
        detection_results: Optional[Dict[str, Any]] = None,
        aoi_polygon: Optional[List[List[float]]] = None,
        aoi_bounds: Optional[List[float]] = None,
        sar_results: Optional[Dict[str, Any]] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Executes deterministic pipeline corresponding to query intent and prepares
        grounded telemetry and evidence items for the response.
        """
        detections = detection_results.get("detections", []) if detection_results else []
        has_npy = bool(sar_results is not None)
        has_image = bool(len(detections) > 0 or has_npy)

        intent = self.classify_intent(query, has_image=has_image, has_npy=has_npy)
        target_classes = self.extract_target_classes(query)

        evidence_items: List[EvidenceItem] = []
        structured_findings: List[str] = []
        spatial_summary: Dict[str, Any] = {}

        # ========================================================
        # 1. COUNT INTENT
        # ========================================================
        if intent == QueryIntent.COUNT:
            if target_classes:
                filtered = [d for d in detections if d.get("class_name", "").lower() in [c.lower() for c in target_classes]]
                count = len(filtered)
                class_label = ", ".join(target_classes[:3])
                structured_findings.append(f"**Verified Target Count:** Exactly `{count}` object(s) matching `{class_label}`.")
                for d in filtered:
                    evidence_items.append(self._detection_to_evidence(d))
            else:
                count = len(detections)
                structured_findings.append(f"**Total Verified Target Count:** Exactly `{count}` object(s) detected across the scene.")
                for d in detections:
                    evidence_items.append(self._detection_to_evidence(d))

        # ========================================================
        # 2. LIST / FILTER INTENT
        # ========================================================
        elif intent in (QueryIntent.LIST, QueryIntent.FILTER):
            filtered = detections
            if target_classes:
                filtered = [d for d in filtered if d.get("class_name", "").lower() in [c.lower() for c in target_classes]]
            
            # Check for confidence filter in query e.g. "> 50%", "highest"
            if "highest" in query.lower():
                if filtered:
                    top_det = max(filtered, key=lambda x: x.get("confidence", 0))
                    filtered = [top_det]
                    structured_findings.append(f"**Highest Confidence Target:** Identified #{top_det['id']} ({top_det['class_name']}) at `{top_det['confidence_percent']}` confidence.")
            
            structured_findings.append(f"**Filtered Targets:** `{len(filtered)}` object(s) matched criteria.")
            for d in filtered[:30]:
                evidence_items.append(self._detection_to_evidence(d))

        # ========================================================
        # 3. SPATIAL / AOI INTENT
        # ========================================================
        elif intent == QueryIntent.SPATIAL:
            aoi_coords = aoi_polygon or aoi_bounds
            if aoi_coords:
                is_latlon = any(d.get("latitude") is not None for d in detections)
                aoi_res = geospatial_service.filter_detections_in_aoi(
                    detections=detections,
                    aoi_coords=aoi_coords,
                    is_latlon=is_latlon
                )
                spatial_summary = aoi_res
                structured_findings.append(
                    f"**Area of Interest Filtering:** `{aoi_res['total_in_aoi']}` target(s) inside AOI, `{aoi_res['total_outside_aoi']}` outside."
                )
                if aoi_res["aoi_area_m2"] > 0:
                    structured_findings.append(f"**AOI Geodesic Area:** `{aoi_res['aoi_area_m2']} m²` ({aoi_res['density_per_sq_km']} targets/km²).")
                for d in aoi_res["matching_detections"]:
                    evidence_items.append(self._detection_to_evidence(d))
            else:
                # General spatial distribution
                structured_findings.append(f"**Spatial Distribution:** Detections span `{detection_results.get('summary', {}).get('resolution', 'Scene')}`.")
                for d in detections[:20]:
                    evidence_items.append(self._detection_to_evidence(d))

        # ========================================================
        # 4. TRAFFIC ANALYSIS
        # ========================================================
        elif intent == QueryIntent.TRAFFIC_ANALYSIS:
            traffic_res = geospatial_service.analyze_traffic_hotspots(detections)
            structured_findings.append(traffic_res["summary_markdown"])
            for d in detections:
                if d.get("id") in traffic_res.get("evidence_ids", []):
                    evidence_items.append(self._detection_to_evidence(d))

        # ========================================================
        # 5. FLOOD ANALYSIS
        # ========================================================
        elif intent == QueryIntent.FLOOD_ANALYSIS:
            water_pct = sar_results.get("coverage_pct") if sar_results else None
            flood_res = multimodal_service.assess_flood_risk(sar_coverage_pct=water_pct)
            structured_findings.append(flood_res["evidence_summary"])

        # ========================================================
        # 6. KNOWLEDGE INTENT (RAG)
        # ========================================================
        elif intent == QueryIntent.KNOWLEDGE:
            if use_rag:
                context, sources = rag.retrieve_documents(query, k=3)
                if sources:
                    structured_findings.append(f"**Retrieved Remote Sensing Knowledge:** (from `{len(sources)}` reference documents).")

        return {
            "intent": intent.value,
            "target_classes": target_classes,
            "structured_findings": "\n\n".join(structured_findings),
            "evidence_items": [e.model_dump() for e in evidence_items],
            "evidence_count": len(evidence_items),
            "spatial_summary": spatial_summary
        }

    def _detection_to_evidence(self, d: Dict[str, Any]) -> EvidenceItem:
        """Converts raw detection dict to structured EvidenceItem."""
        obb = d.get("obb", {})
        cx, cy = obb.get("center_px", [0, 0])
        w = obb.get("width_px", 0)
        h = obb.get("height_px", 0)
        bbox = [cx - w / 2, cy - h / 2, w, h] if w > 0 else [0, 0, 0, 0]

        return EvidenceItem(
            id=d.get("id", 0),
            type="detection",
            label=d.get("class_name", "Target"),
            confidence=d.get("confidence"),
            confidence_percent=d.get("confidence_percent", f"{float(d.get('confidence', 0))*100:.1f}%"),
            center_px=[cx, cy],
            bbox_px=bbox,
            polygon_px=obb.get("polygon_corners_px"),
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            polygon_latlon=obb.get("polygon_corners_latlon"),
            provenance="37-Class YOLO-OBB Detector"
        )


query_router = QueryRouter()
