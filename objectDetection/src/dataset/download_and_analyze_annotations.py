import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_and_analyze():
    api = HfApi()
    repo_id = "blanchon/FAIR1M"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    xml_files = [f for f in repo_files if f.startswith("data/labelXmls/") and f.endswith(".xml")]
    img_files = [f for f in repo_files if f.startswith("data/images/") and (f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png"))]
    
    raw_xml_dir = "data/raw/labelXmls"
    os.makedirs(raw_xml_dir, exist_ok=True)
    os.makedirs("outputs/evaluation", exist_ok=True)
    os.makedirs(".tmp", exist_ok=True)
    
    print(f"Downloading {len(xml_files)} XML annotations...")
    
    def download_xml(xml_path):
        fname = os.path.basename(xml_path)
        dest = os.path.join(raw_xml_dir, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            hf_hub_download(repo_id=repo_id, filename=xml_path, repo_type="dataset", local_dir="data/raw_tmp")
            # move into raw_xml_dir
            tmp_download = os.path.join("data/raw_tmp", xml_path)
            if os.path.exists(tmp_download):
                os.replace(tmp_download, dest)
        return dest

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_xml, x) for x in xml_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading XMLs"):
            f.result()
            
    # Clean up temp if empty
    import shutil
    if os.path.exists("data/raw_tmp"):
        shutil.rmtree("data/raw_tmp", ignore_errors=True)
        
    print(f"Downloaded all XMLs to {raw_xml_dir}")
    
    # Parse all XMLs
    class_counter = Counter()
    image_sizes = Counter()
    total_objects = 0
    malformed_xmls = []
    invalid_coords = []
    stats_per_image = {}
    
    xml_local_files = [os.path.join(raw_xml_dir, f) for f in os.listdir(raw_xml_dir) if f.endswith('.xml')]
    
    for xf in tqdm(xml_local_files, desc="Parsing XML annotations"):
        stem = os.path.splitext(os.path.basename(xf))[0]
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            
            size_elem = root.find("size")
            if size_elem is not None:
                w = float(size_elem.findtext("width", "0"))
                h = float(size_elem.findtext("height", "0"))
                d = int(size_elem.findtext("depth", "3"))
                image_sizes[(w, h, d)] += 1
            else:
                w, h, d = 0, 0, 3
                malformed_xmls.append((xf, "Missing <size> element"))
                
            objs = root.find("objects")
            obj_list = []
            if objs is not None:
                for obj in objs.findall("object"):
                    # Class name
                    poss = obj.find("possibleresult")
                    if poss is not None and poss.find("name") is not None:
                        cname = poss.find("name").text.strip()
                    else:
                        cname = "unknown"
                        
                    points_elem = obj.find("points")
                    pts = []
                    if points_elem is not None:
                        for pt in points_elem.findall("point"):
                            coords = pt.text.strip().split(',')
                            if len(coords) == 2:
                                pts.append((float(coords[0]), float(coords[1])))
                                
                    # Validate points
                    if len(pts) < 4:
                        invalid_coords.append((xf, cname, f"Fewer than 4 points: {len(pts)}"))
                    else:
                        # Check bounding coordinates
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        if min(xs) < -50 or max(xs) > w + 50 or min(ys) < -50 or max(ys) > h + 50:
                            invalid_coords.append((xf, cname, f"Coordinates out of bounds: x=[{min(xs)}, {max(xs)}], y=[{min(ys)}, {max(ys)}] on size {w}x{h}"))
                            
                    class_counter[cname] += 1
                    total_objects += 1
                    obj_list.append({"class": cname, "points": pts})
                    
            stats_per_image[stem] = {
                "width": w,
                "height": h,
                "objects_count": len(obj_list),
            }
            
        except Exception as e:
            malformed_xmls.append((xf, str(e)))
            
    print("\n================== FAIR1M DATASET ANALYSIS ==================")
    print(f"Total XML files parsed: {len(xml_local_files)}")
    print(f"Total objects found: {total_objects}")
    print(f"Total unique classes: {len(class_counter)}")
    print(f"Malformed XML files: {len(malformed_xmls)}")
    print(f"Invalid coordinate objects: {len(invalid_coords)}")
    
    print("\n--- Classes & Object Counts ---")
    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    for cname, count in sorted_classes:
        print(f"  {cname:30s}: {count:6d} ({count/total_objects*100:5.2f}%)")
        
    print("\n--- Image Dimensions Summary ---")
    for dims, count in image_sizes.most_common(10):
        print(f"  {dims[0]} x {dims[1]} x {dims[2]}: {count} images")
        
    report = {
        "total_images_in_hf": len(img_files),
        "total_xmls_in_hf": len(xml_files),
        "total_xmls_parsed": len(xml_local_files),
        "total_objects": total_objects,
        "num_classes": len(class_counter),
        "classes": {c: count for c, count in sorted_classes},
        "image_dimensions": [
            {"width": dims[0], "height": dims[1], "depth": dims[2], "count": count}
            for dims, count in image_sizes.items()
        ],
        "malformed_xmls_count": len(malformed_xmls),
        "invalid_coords_count": len(invalid_coords)
    }
    
    with open("outputs/evaluation/dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Write class map YAML/JSON for reference
    class_list = [c for c, _ in sorted_classes]
    class_map = {i: c for i, c in enumerate(class_list)}
    with open("data/class_mapping.json", "w") as f:
        json.dump({"class_to_id": {c: i for i, c in enumerate(class_list)}, "id_to_class": class_map}, f, indent=2)
        
    print(f"\nReport saved to outputs/evaluation/dataset_report.json")
    print(f"Class mapping saved to data/class_mapping.json")

if __name__ == "__main__":
    download_and_analyze()
