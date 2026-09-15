import os
import torch
from ultralytics import YOLO

def test_obb_support():
    print("Testing Ultralytics YOLO-OBB capabilities...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target device: {device}")
    
    # Try loading yolo11n-obb.pt (or yolov8n-obb.pt)
    for model_name in ["yolo11n-obb.pt", "yolov8n-obb.pt"]:
        try:
            print(f"\nAttempting to load {model_name}...")
            model = YOLO(model_name)
            print(f"Successfully loaded {model_name}!")
            print(f"Task: {model.task}")
            assert model.task == "obb", f"Expected task 'obb', got {model.task}"
            
            # Test dummy prediction
            dummy_img = torch.zeros((3, 640, 640), dtype=torch.uint8).numpy()
            results = model.predict(dummy_img, device=device, verbose=False)
            print(f"Prediction successful on {device}!")
            print(f"Result OBB attribute exists: {hasattr(results[0], 'obb')}")
            return model_name
        except Exception as e:
            print(f"Error loading/testing {model_name}: {e}")
            
    return None

if __name__ == "__main__":
    best_model = test_obb_support()
    print(f"\nResult: Best supported OBB model = {best_model}")
