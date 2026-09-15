import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportService:
    """
    Automated Earth Observation Intelligence Report Generator.
    Assembles structured telemetry, model detections, spatial metrics,
    and decision-support assessments into an executive intelligence report.
    """

    def generate_report(
        self,
        title: str = "VisionOrbit Earth Observation Intelligence Report",
        aoi_name: str = "Target AOI Footprint",
        detections: List[Dict[str, Any]] = [],
        sar_telemetry: Optional[Dict[str, Any]] = None,
        change_telemetry: Optional[Dict[str, Any]] = None,
        traffic_telemetry: Optional[Dict[str, Any]] = None,
        flood_telemetry: Optional[Dict[str, Any]] = None,
        sensor_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds Markdown and HTML report documents from verified telemetry.
        """
        report_id = f"SATREP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        sensor_name = sensor_info.get("sensor", "Multi-Sensor Optical / Sentinel-1 SAR") if sensor_info else "Multi-Sensor Optical / Sentinel-1 SAR"
        resolution_str = sensor_info.get("resolution", "Sub-meter High-Res Aerial / 10m Sentinel") if sensor_info else "Sub-meter High-Res Aerial / 10m Sentinel"
        crs_str = sensor_info.get("crs", "WGS84 / EPSG:4326") if sensor_info else "WGS84 / EPSG:4326"

        # 1. Target Breakdown
        from collections import Counter
        class_counts = dict(Counter(d.get("class_name", "Unknown") for d in detections))
        total_dets = len(detections)
        confs = [d.get("confidence", 0) * 100 for d in detections]
        avg_conf = f"{sum(confs)/len(confs):.1f}%" if confs else "N/A"

        # Build Markdown Document
        md_lines = []
        md_lines.append(f"# 🛰️ {title}")
        md_lines.append(f"**Report ID:** `{report_id}` | **Generated:** `{timestamp_str}` | **Classification:** `OPERATIONAL INTEL`\n")
        md_lines.append("---")
        
        md_lines.append("## 1. Mission & Area of Interest (AOI) Telemetry")
        md_lines.append(f"| Parameter | Value |")
        md_lines.append(f"| :--- | :--- |")
        md_lines.append(f"| **Target Location / AOI** | `{aoi_name}` |")
        md_lines.append(f"| **Sensor & Constellation** | `{sensor_name}` |")
        md_lines.append(f"| **Coordinate Reference System (CRS)** | `{crs_str}` |")
        md_lines.append(f"| **Effective Resolution / GSD** | `{resolution_str}` |")
        md_lines.append(f"| **Total Verified Targets** | `{total_dets}` |")
        md_lines.append(f"| **Mean Model Confidence** | `{avg_conf}` |\n")

        md_lines.append("## 2. Target Inventory & Classification Breakdown")
        if total_dets > 0:
            md_lines.append("| Object Class | Detected Count | Share (%) | Model Engine |")
            md_lines.append("| :--- | :--- | :--- | :--- |")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                pct = round(cnt / total_dets * 100, 1)
                md_lines.append(f"| **{cls}** | `{cnt}` | `{pct}%` | `37-Class YOLO-OBB` |")
        else:
            md_lines.append("*No targets identified exceeding detection confidence thresholds.*\n")

        # Key Detections Table
        if detections:
            md_lines.append("\n### Key Target Coordinates (Top Detections)")
            md_lines.append("| # | Class | Confidence | Coordinates (Lat, Lon) | Center (Pixel Space) |")
            md_lines.append("|---|---|---|---|---|")
            for d in detections[:15]:
                lat_lon = f"({d['latitude']:.6f}, {d['longitude']:.6f})" if d.get("latitude") is not None else "Pixel Space"
                cx, cy = d.get("obb", {}).get("center_px", [0, 0])
                md_lines.append(f"| #{d.get('id', '-')} | **{d.get('class_name')}** | `{d.get('confidence_percent', 'N/A')}` | `{lat_lon}` | `({int(cx)}, {int(cy)})` |")
            if len(detections) > 15:
                md_lines.append(f"\n*... and {len(detections) - 15} additional validated targets.*")

        # 3. Decision Support Modules
        md_lines.append("\n## 3. Decision-Support & Risk Telemetry")
        
        if traffic_telemetry:
            md_lines.append(f"### 🚦 Traffic Density & Hotspot Assessment")
            md_lines.append(f"- **Risk Level:** `{traffic_telemetry.get('risk_level', 'Nominal')}` ({traffic_telemetry.get('confidence', 'Moderate')})")
            md_lines.append(f"- **Total Vehicles Identified:** `{traffic_telemetry.get('total_vehicles', 0)}`")
            md_lines.append(f"- **Spatial Clusters:** `{traffic_telemetry.get('hotspot_clusters_count', 0)}` vehicle aggregations.")
            md_lines.append(f"- **Density Ratio:** `{traffic_telemetry.get('vehicle_density_ratio', 0)}` vehicles/Megapixel.\n")

        if flood_telemetry:
            md_lines.append(f"### 🌊 Flood & Inundation Risk Indicator")
            md_lines.append(f"- **Risk Level:** `{flood_telemetry.get('risk_level', 'Nominal')}` ({flood_telemetry.get('confidence', 'Moderate')})")
            md_lines.append(f"- **Water Extent Coverage:** `{flood_telemetry.get('water_extent_pct', 0)}%`")
            md_lines.append(f"- **Terrain Vulnerability:** `{flood_telemetry.get('terrain_vulnerability', 'Moderate')}`.\n")

        if sar_telemetry:
            md_lines.append(f"### 📡 Synthetic Aperture Radar (SAR) Telemetry")
            md_lines.append(f"- **Mean Radar Backscatter:** `{sar_telemetry.get('mean_backscatter_db', 'N/A')} dB`")
            md_lines.append(f"- **Segmentation Regions:** `{sar_telemetry.get('detected_regions_count', 0)}` connected radar features.")
            md_lines.append(f"- **Surface Coverage:** `{sar_telemetry.get('coverage_pct', 0)}%`.\n")

        if change_telemetry:
            md_lines.append(f"### 🔄 Multi-Temporal Change Telemetry")
            md_lines.append(f"- **Total Changed Area:** `{change_telemetry.get('changed_pixels', 0)} px` (`{change_telemetry.get('percentage_change', 0)}%` scene variance).")
            md_lines.append(f"- **Identified Change Clusters:** `{change_telemetry.get('changed_regions_count', 0)}`.\n")

        # 4. Provenance & Limitations
        md_lines.append("## 4. Provenance & Scientific Guardrails")
        md_lines.append("- **AI Engines:** 37-Class YOLO-OBB Detector, 7-Channel Sentinel CNN U-Net, Deterministic Shapely/GeoPandas GIS Engine.")
        md_lines.append("- **Verification Standard:** All numerical counts and geographic metrics are calculated directly via deterministic algorithms.")
        md_lines.append("- **Operational Disclaimer:** Satellite indicators provide strategic observation and decision support; operational deployments should be corroborated with in-situ field telemetry.")

        markdown_content = "\n".join(md_lines)

        # Generate Styled HTML for preview / printing
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} - {report_id}</title>
<style>
  body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; padding: 40px; margin: 0; }}
  .report-card {{ max-width: 850px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
  .header-row {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0284c7; padding-bottom: 20px; margin-bottom: 30px; }}
  h1 {{ color: #0f172a; margin: 0 0 8px 0; font-size: 24px; }}
  .badge {{ background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0 25px 0; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f1f5f9; color: #334155; font-weight: 600; }}
  h2 {{ color: #0369a1; font-size: 18px; margin-top: 25px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  .disclaimer {{ background: #f8fafc; border-left: 4px solid #0284c7; padding: 12px; font-size: 13px; color: #64748b; margin-top: 30px; border-radius: 0 8px 8px 0; }}
  @media print {{ body {{ background: #fff; padding: 0; }} .report-card {{ box-shadow: none; border: none; padding: 0; }} }}
</style>
</head>
<body>
<div class="report-card">
  <div class="header-row">
    <div>
      <h1>🛰️ {title}</h1>
      <div style="color: #64748b; font-size: 14px;"><strong>ID:</strong> {report_id} • <strong>Date:</strong> {timestamp_str}</div>
      <div style="color: #64748b; font-size: 14px;"><strong>AOI:</strong> {aoi_name} • <strong>CRS:</strong> {crs_str}</div>
    </div>
    <span class="badge">Operational Intel</span>
  </div>

  <h2>1. Mission & Sensor Telemetry</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Target Footprint</td><td>{aoi_name}</td></tr>
    <tr><td>Sensor Platform</td><td>{sensor_name}</td></tr>
    <tr><td>Spatial Resolution</td><td>{resolution_str}</td></tr>
    <tr><td>Total Verified Targets</td><td><strong>{total_dets}</strong></td></tr>
    <tr><td>Mean Confidence</td><td>{avg_conf}</td></tr>
  </table>

  <h2>2. Target Classification Inventory</h2>
  <table>
    <tr><th>Class Name</th><th>Count</th><th>Share (%)</th><th>Engine</th></tr>
    {"".join([f"<tr><td><strong>{c}</strong></td><td>{n}</td><td>{round(n/total_dets*100, 1)}%</td><td>YOLO-OBB</td></tr>" for c, n in class_counts.items()]) if total_dets else "<tr><td colspan='4'>No targets detected</td></tr>"}
  </table>

  <div class="disclaimer">
    <strong>Verification Standard:</strong> SatQuery AI deterministic GIS and deep learning verification pipeline. Numerical assertions are validated by bounding box geometries and spatial index arrays.
  </div>
</div>
</body>
</html>
"""

        return {
            "report_id": report_id,
            "created_at": timestamp_str,
            "title": title,
            "markdown_content": markdown_content,
            "html_content": html_content,
            "summary": {
                "total_detections": total_dets,
                "class_breakdown": class_counts,
                "average_confidence": avg_conf,
                "aoi": aoi_name
            },
            "provenance": {
                "report_id": report_id,
                "generator": "VisionOrbit Intelligence Report Engine v2.0",
                "timestamp": timestamp_str
            }
        }


report_service = ReportService()
