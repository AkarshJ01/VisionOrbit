import os
import argparse
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_dataset(limit=None):
    api = HfApi()
    repo_id = "blanchon/FAIR1M"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    img_files = sorted([f for f in repo_files if f.startswith("data/images/") and (f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png"))])
    xml_files = sorted([f for f in repo_files if f.startswith("data/labelXmls/") and f.endswith(".xml")])
    
    img_dir = "data/raw/images"
    xml_dir = "data/raw/labelXmls"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    
    if limit is not None and limit > 0:
        img_files = img_files[:limit]
        # find matching xmls
        stems = set(os.path.splitext(os.path.basename(f))[0] for f in img_files)
        xml_files = [f for f in xml_files if os.path.splitext(os.path.basename(f))[0] in stems]
        
    print(f"Target download: {len(img_files)} images, {len(xml_files)} XML annotations...")
    
    # Download XMLs
    def download_single(file_path, dest_dir):
        fname = os.path.basename(file_path)
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        try:
            hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", local_dir="data/.hf_cache")
            cached = os.path.join("data/.hf_cache", file_path)
            if os.path.exists(cached):
                os.replace(cached, dest)
            return dest
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")
            return None

    # Download XMLs first
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_single, xf, xml_dir) for xf in xml_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading XML annotations"):
            f.result()
            
    # Download Images
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single, img, img_dir) for img in img_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading Images"):
            f.result()
            
    # Clean cache if empty
    import shutil
    if os.path.exists("data/.hf_cache"):
        shutil.rmtree("data/.hf_cache", ignore_errors=True)
        
    print(f"Download complete. Stored in {img_dir} and {xml_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FAIR1M dataset from HuggingFace")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to download")
    args = parser.parse_args()
    download_dataset(limit=args.limit)
