import os
import shutil
import json
import argparse
import torch
from ultralytics import YOLO

def evaluate(model_path="models/best.pt", data_yaml="data/fair1m_obb.yaml", split="val", imgsz=640, batch=8, device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    os.makedirs("outputs/evaluation", exist_ok=True)
    
    print("==================================================")
    print("       SatQuery Model Evaluation Pipeline         ")
    print("==================================================")
    print(f"Model path:  {model_path}")
    print(f"Data config: {data_yaml}")
    print(f"Split:       {split}")
    print(f"Device:      {device}")
    print("==================================================\n")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=True,
        plots=True,
        verbose=True
    )
    
    # Extract overall metrics
    results_summary = {
        "model": model_path,
        "split": split,
        "device": device,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {}
    }
    
    # Extract per-class metrics
    class_names = model.names
    for idx, cname in class_names.items():
        if idx < len(metrics.box.p):
            p = float(metrics.box.p[idx])
            r = float(metrics.box.r[idx])
            map50 = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.0
            map_all = float(metrics.box.ap[idx]) if idx < len(metrics.box.ap) else 0.0
            results_summary["per_class"][cname] = {
                "class_id": int(idx),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map_all, 4)
            }
            
    # Copy generated plots to outputs/evaluation
    save_dir = str(metrics.save_dir)
    if os.path.exists(save_dir):
        for fname in os.listdir(save_dir):
            if fname.endswith(('.png', '.jpg', '.csv', '.json')):
                shutil.copy2(os.path.join(save_dir, fname), os.path.join("outputs/evaluation", fname))
                
    out_json = "outputs/evaluation/evaluation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Split:    {split.upper()}")
    print(f"Precision (P): {results_summary['precision']:.4f}")
    print(f"Recall (R):    {results_summary['recall']:.4f}")
    print(f"mAP@50:        {results_summary['mAP50']:.4f}")
    print(f"mAP@50-95:     {results_summary['mAP50_95']:.4f}")
    print(f"Saved evaluation metrics to: {out_json}")
    
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SatQuery YOLO-OBB Model")
    parser.add_argument("--model", type=str, default="models/best.pt")
    parser.add_argument("--data", type=str, default="data/fair1m_obb.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
