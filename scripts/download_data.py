#!/usr/bin/env python
"""
Download FG-Diff datasets from HuggingFace.
Usage: python scripts/download_data.py
"""

import os
from huggingface_hub import snapshot_download

# Set mirror for China users (can also be set via environment variable)
# export HF_ENDPOINT=https://hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = "ModelsWeights/AD-FG-Diff"
REPO_TYPE = "dataset"
LOCAL_DIR = "./data"


def download_datasets():
    """Download all datasets from HuggingFace."""
    print(f"Downloading datasets from {REPO_ID}...")
    print(f"Using HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Download entire dataset repository
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=LOCAL_DIR,
                resume_download=True,  # Resume if interrupted
                max_workers=4,  # Limit concurrent downloads
            )
            print(f"Successfully downloaded datasets to {LOCAL_DIR}")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("\nDownload failed after all retries.")
                print("Try setting the mirror manually:")
                print("  export HF_ENDPOINT=https://hf-mirror.com")
                print("  python scripts/download_data.py")
                raise


if __name__ == "__main__":
    download_datasets()
