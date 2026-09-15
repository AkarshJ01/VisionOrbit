import os
import sys
import time
import json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.tiled_inference import TiledOBBInferrer
from src.inference.predict import load_image_rgb

def benchmark(model_path="models/best.pt", test_dir="data/processed/images/test", num_runs=20):
    os.makedirs("outputs/evaluation", exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("==================================================")
    print("        SatQuery Inference Speed Benchmark        ")
    print("==================================================")
    print(f"Model:   {model_path}")
    print(f"Device:  {device}")
    print(f"Runs:    {num_runs}")
    print("==================================================\n")
    
    inferrer = TiledOBBInferrer(model_path=model_path, device=device)
    
    # Load sample image
    sample_img = None
    if os.path.exists(test_dir):
        files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        if files:
            sample_img = load_image_rgb(files[0])
            
    if sample_img is None:
        sample_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
    h, w = sample_img.shape[:2]
    print(f"Benchmarking on {w}x{h} image canvas...")
    
    # Warmup
    print("Warming up GPU/MPS...")
    for _ in range(3):
        _ = inferrer.predict_image(sample_img, conf_threshold=0.25, use_tiling=False)
        
    # Standard Inference Latency
    print("\nBenchmarking Standard Direct Inference (640px resized)...")
    latencies_direct = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = inferrer.predict_image(sample_img, conf_threshold=0.25, use_tiling=False)
        latencies_direct.append((time.perf_counter() - t0) * 1000) # ms
        
    avg_direct_ms = np.mean(latencies_direct)
    fps_direct = 1000.0 / avg_direct_ms
    print(f"Standard Inference: {avg_direct_ms:.2f} ms/image ({fps_direct:.1f} FPS)")
    
    # Tiled Inference Latency
    print("\nBenchmarking Tiled Sliding-Window Inference (640px tiles + NMS)...")
    latencies_tiled = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = inferrer.predict_image(sample_img, conf_threshold=0.25, use_tiling=True)
        latencies_tiled.append((time.perf_counter() - t0) * 1000) # ms
        
    avg_tiled_ms = np.mean(latencies_tiled)
    fps_tiled = 1000.0 / avg_tiled_ms
    print(f"Tiled Inference:    {avg_tiled_ms:.2f} ms/image ({fps_tiled:.1f} FPS)")
    
    bench_data = {
        "device": device,
        "image_size": f"{w}x{h}",
        "num_runs": num_runs,
        "standard_inference": {
            "avg_latency_ms": round(float(avg_direct_ms), 2),
            "min_latency_ms": round(float(np.min(latencies_direct)), 2),
            "max_latency_ms": round(float(np.max(latencies_direct)), 2),
            "fps": round(float(fps_direct), 1)
        },
        "tiled_inference": {
            "avg_latency_ms": round(float(avg_tiled_ms), 2),
            "min_latency_ms": round(float(np.min(latencies_tiled)), 2),
            "max_latency_ms": round(float(np.max(latencies_tiled)), 2),
            "fps": round(float(fps_tiled), 1)
        }
    }
    
    out_file = "outputs/evaluation/benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(bench_data, f, indent=2)
        
    print(f"\nBenchmark results saved to: {out_file}")
    return bench_data

if __name__ == "__main__":
    benchmark()
