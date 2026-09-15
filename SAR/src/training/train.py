import os
import shutil
import json
import argparse
import torch
from ultralytics import YOLO

def train(data_yaml="data/fair1m_obb.yaml",
          model_name="yolo11n-obb.pt",
          epochs=15,
          imgsz=640,
          batch=8,
          device=None,
          workers=2,
          project="runs/train",
          name="satquery_obb",
          resume=False,
          smoke_test=False):
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/evaluation", exist_ok=True)
    os.makedirs(".tmp", exist_ok=True)
    
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    print("==================================================")
    print("        SatQuery YOLO-OBB Training Pipeline        ")
    print("==================================================")
    print(f"Data config: {data_yaml}")
    print(f"Base model:  {model_name}")
    print(f"Target device: {device}")
    print(f"Image size:  {imgsz}")
    print(f"Batch size:  {batch}")
    print(f"Epochs:      {1 if smoke_test else epochs}")
    print(f"Workers:     {workers}")
    print(f"Smoke test:  {smoke_test}")
    print("==================================================\n")
    
    if smoke_test:
        epochs = 1
        name = "smoke_test"
        
    # Initialize YOLO-OBB model
    model = YOLO(model_name)
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        plots=True,
        save=True,
        save_period=1,
        patience=10,
        exist_ok=True,
        verbose=True
    )
    
    # Obtain actual save directory from trainer or results
    run_dir = str(results.save_dir) if hasattr(results, 'save_dir') else (
        str(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else os.path.join(project, name)
    )
    
    weights_dir = os.path.join(run_dir, "weights")
    best_weight = os.path.join(weights_dir, "best.pt")
    last_weight = os.path.join(weights_dir, "last.pt")
    
    saved_weight = best_weight if os.path.exists(best_weight) else last_weight
    target_model_path = "models/best.pt"
    
    if os.path.exists(saved_weight):
        shutil.copy2(saved_weight, target_model_path)
        print(f"\n[OK] Model successfully saved to {target_model_path}")
    else:
        print(f"\n[Warning] No weight file found in {weights_dir}")
        
    # Copy evaluation plots and results to outputs/evaluation
    if os.path.exists(run_dir):
        for fname in os.listdir(run_dir):
            if fname.endswith(('.png', '.csv', '.jpg')):
                src = os.path.join(run_dir, fname)
                dst = os.path.join("outputs/evaluation", fname)
                shutil.copy2(src, dst)
                
    # Record training metrics
    metrics_summary = {
        "status": "success",
        "smoke_test": smoke_test,
        "model": model_name,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch,
        "best_model_path": target_model_path,
        "results_dir": run_dir
    }
    
    if hasattr(results, 'results_dict') and results.results_dict:
        metrics_summary["metrics"] = results.results_dict
        
    with open("outputs/evaluation/training_results.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Training summary saved to outputs/evaluation/training_results.json")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatQuery YOLO-OBB on FAIR1M")
    parser.add_argument("--data", type=str, default="data/fair1m_obb.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume previous run")
    
    args = parser.parse_args()
    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        smoke_test=args.smoke_test,
        resume=args.resume
    )
