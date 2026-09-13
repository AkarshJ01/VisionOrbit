import os
from huggingface_hub import HfApi, list_repo_files

def inspect_fair1m_repo():
    api = HfApi()
    repo_id = "blanchon/FAIR1M"
    print(f"Inspecting HuggingFace repo: {repo_id}...")
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"Total files found in repo: {len(files)}")
        print("\nSample files:")
        for f in files[:30]:
            print(f"  {f}")
        if len(files) > 30:
            print(f"  ... and {len(files)-30} more")
        
        # Check extensions and directories
        dirs = set(os.path.dirname(f) for f in files if os.path.dirname(f))
        print("\nDirectories in repo:")
        for d in sorted(dirs)[:20]:
            print(f"  {d}")
            
        exts = set(os.path.splitext(f)[1] for f in files)
        print(f"\nFile extensions: {exts}")
        
    except Exception as e:
        print(f"Error inspecting repo: {e}")

if __name__ == "__main__":
    inspect_fair1m_repo()
