"""
Step 1: Download the Kaggle Chest X-Ray Pneumonia dataset.

Usage:
    python scripts/01_download_data.py

Credentials are loaded from .env file in the project root, environment variables,
or ~/.kaggle/kaggle.json.
"""

import os
import shutil
from pathlib import Path

# Load .env file
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: kaggle\n"
        "Install it with: python -m pip install -r requirements.txt"
    ) from exc

# These are the temporary Kaggle folder and the one final dataset folder.
# Dataset paths
DATA_DIR = Path("data/raw").resolve()
DOWNLOAD_DIR = DATA_DIR / "kaggle_download"
FINAL_DIR = DATA_DIR / "chest_xray"
DATASET = "paultimothymooney/chest-xray-pneumonia"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Chest X-Ray Pneumonia dataset from Kaggle...")
# Kaggle downloads one dataset, then this code finds and moves the real chest_xray folder.
# Kaggle download and extraction
api = KaggleApi()
api.authenticate()
api.dataset_download_files(DATASET, path=str(DOWNLOAD_DIR), unzip=True, quiet=False)

candidate_dirs = [
    DOWNLOAD_DIR / "chest_xray",
    DOWNLOAD_DIR / "chest-xray-pneumonia" / "chest_xray",
    DOWNLOAD_DIR / "chest-xray-pneumonia",
]

source_dir = next((path for path in candidate_dirs if path.exists()), None)
if source_dir is None:
    raise SystemExit(
        f"Download finished, but the expected chest_xray folder was not found in {DOWNLOAD_DIR}"
    )

if FINAL_DIR.exists():
    print(f"Dataset already exists at: {FINAL_DIR}")
else:
    shutil.move(str(source_dir), str(FINAL_DIR))
    print(f"Moved dataset to: {FINAL_DIR}")

# End by checking that every split and class exists and contains images.
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = FINAL_DIR / split / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.jpeg")))
            print(f"  {split}/{cls}: {count} images")
        else:
            print(f"  {split}/{cls}: NOT FOUND - check the extracted folder structure")

print("\nStep 1 outputs")
print(f"  Output directory: {FINAL_DIR}")
print(f"  Train images:     {FINAL_DIR / 'train'}")
print(f"  Validation images:{FINAL_DIR / 'val'}")
print(f"  Test images:      {FINAL_DIR / 'test'}")
print(f"  Download cache:   {DOWNLOAD_DIR}")
print("\nIf the folder structure looks wrong, manually move so it looks like:")
print("  data/raw/chest_xray/train/NORMAL/")
print("  data/raw/chest_xray/train/PNEUMONIA/")
print("  data/raw/chest_xray/val/...")
print("  data/raw/chest_xray/test/...")
