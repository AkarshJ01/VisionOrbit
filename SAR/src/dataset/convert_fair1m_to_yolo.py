import os
import shutil
import json
import random
import yaml
import cv2
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_class_mappings():
    mapping_file = "data/class_mapping.json"
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            data = json.load(f)
            return data["class_to_id"], data["id_to_class"]
            
    classes = [
        "Small Car", "Van", "Dump Truck", "Cargo Truck", "Dry Cargo Ship",
        "Motorboat", "Intersection", "other-vehicle", "Fishing Boat", "other-ship",
        "other-airplane", "A220", "Liquid Cargo Ship", "Tennis Court", "Boeing737",
        "Tugboat", "Bus", "Passenger Ship", "Engineering Ship", "A321",
        "Excavator", "Trailer", "Truck Tractor", "Baseball Field", "Football Field",
        "Warship", "Basketball Court", "Tractor", "A330", "Boeing787",
        "Bridge", "Boeing747", "Boeing777", "ARJ21", "A350", "Roundabout", "C919"
    ]
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {str(i): c for i, c in enumerate(classes)}
    os.makedirs("data", exist_ok=True)
    with open(mapping_file, "w") as f:
        json.dump({"class_to_id": class_to_id, "id_to_class": id_to_class}, f, indent=2)
    return class_to_id, id_to_class

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, val))

def read_image_rgb(src_path):
    """Robustly read any image/TIFF format and return a 3-channel uint8 RGB numpy array."""
    if src_path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(src_path)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] >= 3:
                    img = img[:, :, :3] # Discard alpha / extra bands
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            return img
        except Exception:
            pass
            
    # Fallback to OpenCV
    img = cv2.imread(src_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Fallback to PIL
    with Image.open(src_path) as pimg:
        return np.array(pimg.convert("RGB"))

def parse_xml_annotation(xml_path, class_to_id, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find("size")
    if size_elem is not None:
        width = float(size_elem.findtext("width", str(img_w)))
        height = float(size_elem.findtext("height", str(img_h)))
    else:
        width, height = float(img_w), float(img_h)
        
    if width <= 0 or height <= 0:
        width, height = float(img_w), float(img_h)
        
    objs = root.find("objects")
    obb_lines = []
    skipped_count = 0
    
    if objs is not None:
        for obj in objs.findall("object"):
            poss = obj.find("possibleresult")
            if poss is not None and poss.find("name") is not None:
                cname = poss.find("name").text.strip()
            else:
                cname = "unknown"
                
            if cname not in class_to_id:
                skipped_count += 1
                continue
                
            cid = class_to_id[cname]
            
            points_elem = obj.find("points")
            pts = []
            if points_elem is not None:
                for pt in points_elem.findall("point"):
                    coords = pt.text.strip().split(',')
                    if len(coords) == 2:
                        pts.append((float(coords[0]), float(coords[1])))
                        
            if len(pts) >= 4:
                p1, p2, p3, p4 = pts[0], pts[1], pts[2], pts[3]
                
                # Normalize and clamp coordinates to [0, 1]
                x1, y1 = clamp(p1[0] / width), clamp(p1[1] / height)
                x2, y2 = clamp(p2[0] / width), clamp(p2[1] / height)
                x3, y3 = clamp(p3[0] / width), clamp(p3[1] / height)
                x4, y4 = clamp(p4[0] / width), clamp(p4[1] / height)
                
                if (x1 == x2 == x3 == x4) and (y1 == y2 == y3 == y4):
                    skipped_count += 1
                    continue
                    
                line = f"{cid} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                obb_lines.append(line)
            else:
                skipped_count += 1
                
    return width, height, obb_lines, skipped_count

def convert_dataset(seed=42):
    random.seed(seed)
    class_to_id, id_to_class = get_class_mappings()
    
    raw_img_dir = "data/raw/images"
    raw_xml_dir = "data/raw/labelXmls"
    processed_dir = "data/processed"
    
    # Remove existing cache files
    for cache_f in Path(processed_dir).rglob("*.cache"):
        os.remove(cache_f)
        
    if not os.path.exists(raw_img_dir) or not os.path.exists(raw_xml_dir):
        raise FileNotFoundError(f"Missing raw data in {raw_img_dir} or {raw_xml_dir}")
        
    img_files = sorted([f for f in os.listdir(raw_img_dir) if f.endswith(('.tif', '.jpg', '.png'))])
    
    valid_pairs = []
    for img_name in img_files:
        stem = os.path.splitext(img_name)[0]
        xml_name = f"{stem}.xml"
        xml_path = os.path.join(raw_xml_dir, xml_name)
        if os.path.exists(xml_path):
            valid_pairs.append((img_name, xml_name, stem))
            
    print(f"Total valid image-XML pairs available: {len(valid_pairs)}")
    if len(valid_pairs) == 0:
        print("No image-XML pairs found!")
        return
        
    random.shuffle(valid_pairs)
    
    n_total = len(valid_pairs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    
    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train+n_val],
        "test": valid_pairs[n_train+n_val:]
    }
    
    print(f"Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    total_objects_converted = 0
    total_objects_skipped = 0
    
    for split_name, pairs in splits.items():
        img_out_dir = os.path.join(processed_dir, "images", split_name)
        lbl_out_dir = os.path.join(processed_dir, "labels", split_name)
        
        # Clean previous images in directory if re-converting
        shutil.rmtree(img_out_dir, ignore_errors=True)
        shutil.rmtree(lbl_out_dir, ignore_errors=True)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for img_name, xml_name, stem in tqdm(pairs, desc=f"Processing {split_name} split"):
            src_img = os.path.join(raw_img_dir, img_name)
            src_xml = os.path.join(raw_xml_dir, xml_name)
            
            # Standardize output image as 3-channel RGB JPEG for high performance & uniform shape
            dst_img = os.path.join(img_out_dir, f"{stem}.jpg")
            dst_lbl = os.path.join(lbl_out_dir, f"{stem}.txt")
            
            img_rgb = read_image_rgb(src_img)
            h, w = img_rgb.shape[:2]
            
            # Save 3-channel RGB image
            Image.fromarray(img_rgb).save(dst_img, quality=95)
            
            _, _, obb_lines, skipped = parse_xml_annotation(src_xml, class_to_id, w, h)
            total_objects_converted += len(obb_lines)
            total_objects_skipped += skipped
            
            with open(dst_lbl, "w") as f:
                for line in obb_lines:
                    f.write(line + "\n")
                    
    # Generate YOLO dataset YAML
    workspace_root = os.path.abspath(os.getcwd())
    yaml_content = {
        "path": os.path.join(workspace_root, "data/processed"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in id_to_class.items()}
    }
    
    yaml_path = "data/fair1m_obb.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
    print(f"\n================ Conversion Complete ================")
    print(f"Total Objects Converted: {total_objects_converted}")
    print(f"Total Objects Skipped/Filtered: {total_objects_skipped}")
    print(f"Dataset YAML written to: {yaml_path}")

if __name__ == "__main__":
    convert_dataset()
