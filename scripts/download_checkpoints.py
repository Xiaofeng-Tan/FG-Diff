#!/usr/bin/env python
"""
Download FG-Diff pre-trained checkpoints from HuggingFace.
Usage: python scripts/download_checkpoints.py
"""

import os
from huggingface_hub import snapshot_download

# Set mirror for China users (can also be set via environment variable)
# export HF_ENDPOINT=https://hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = "ModelsWeights/AD-FG-Diff"
REPO_TYPE = "model"
LOCAL_DIR = "./checkpoints"


def download_checkpoints():
    """Download all checkpoints from HuggingFace."""
    print(f"Downloading checkpoints from {REPO_ID}...")
    print(f"Using HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    
    try:
        # Download entire model repository
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded checkpoints to {LOCAL_DIR}")
    except Exception as e:
        print(f"Error downloading checkpoints: {e}")
        print("\nTry setting the mirror manually:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        print("  python scripts/download_checkpoints.py")
        raise


if __name__ == "__main__":
    download_checkpoints()
