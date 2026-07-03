"""
Step 1: Download the Kaggle Chest X-Ray Pneumonia dataset.

Usage:
    python scripts/01_download_data.py

Credentials are loaded from .env file in the project root.
"""

import os
from pathlib import Path

# Load .env file
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

import opendatasets as od

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"

print("Downloading Chest X-Ray Pneumonia dataset from Kaggle...")
od.download(DATASET_URL, data_dir=str(DATA_DIR))

expected_dir = DATA_DIR / "chest-xray-pneumonia" / "chest_xray"
if not expected_dir.exists():
    expected_dir = DATA_DIR / "chest-xray-pneumonia"

for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = expected_dir / split / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.jpeg")))
            print(f"  {split}/{cls}: {count} images")
        else:
            print(f"  {split}/{cls}: NOT FOUND — check the extracted folder structure")

print(f"\nDataset saved to: {expected_dir}")
print("If the folder structure looks wrong, manually move so it looks like:")
print("  data/raw/chest_xray/train/NORMAL/")
print("  data/raw/chest_xray/train/PNEUMONIA/")
print("  data/raw/chest_xray/val/...")
print("  data/raw/chest_xray/test/...")
