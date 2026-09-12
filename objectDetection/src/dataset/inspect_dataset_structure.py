import os
import xml.etree.ElementTree as ET
from huggingface_hub import HfApi, hf_hub_download
from collections import Counter
from PIL import Image

def inspect_dataset_structure():
    api = HfApi()
    repo_id = "blanchon/FAIR1M"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    images = [f for f in repo_files if f.startswith("data/images/") and (f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png"))]
    xmls = [f for f in repo_files if f.startswith("data/labelXmls/") and f.endswith(".xml")]
    
    print(f"Total image files in repo: {len(images)}")
    print(f"Total XML files in repo: {len(xmls)}")
    
    # Check matching basenames
    img_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in images}
    xml_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in xmls}
    
    matched = set(img_stems.keys()).intersection(set(xml_stems.keys()))
    print(f"Matched image-XML pairs: {len(matched)}")
    print(f"Images without XML: {len(set(img_stems.keys()) - set(xml_stems.keys()))}")
    print(f"XMLs without images: {len(set(xml_stems.keys()) - set(img_stems.keys()))}")
    
    # Download 5 sample XMLs and 2 sample images to .tmp/sample_inspect
    sample_dir = ".tmp/sample_inspect"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_stems = sorted(list(matched))[:5]
    print(f"\nDownloading sample annotations for stems: {sample_stems}...")
    
    for stem in sample_stems:
        xml_file = xml_stems[stem]
        img_file = img_stems[stem]
        local_xml = hf_hub_download(repo_id=repo_id, filename=xml_file, repo_type="dataset", local_dir=sample_dir)
        print(f"Downloaded XML: {local_xml}")
        if stem in sample_stems[:2]:
            local_img = hf_hub_download(repo_id=repo_id, filename=img_file, repo_type="dataset", local_dir=sample_dir)
            print(f"Downloaded Image: {local_img}")
            
    # Inspect downloaded XML structure
    for stem in sample_stems:
        xml_path = os.path.join(sample_dir, xml_stems[stem])
        print(f"\n--- XML Content for {stem}.xml ---")
        with open(xml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("".join(lines[:40]))
            if len(lines) > 40:
                print(f"... ({len(lines)-40} more lines)")

if __name__ == "__main__":
    inspect_dataset_structure()
