"""
Step 2: Explore and preprocess the dataset.

Usage:
    python scripts/02_preprocess.py

What it does:
    - Counts images per class per split
    - Computes image size statistics
    - Computes pixel intensity stats per class
    - Saves a summary JSON to experiments/results/data_summary.json
    - Saves sample images grid to experiments/results/sample_images.png
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/raw/chest_xray")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# --- Count images ---
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

summary = {}

for split in ["train", "val", "test"]:
    summary[split] = {}
    for cls in CLASS_NAMES:
        cls_dir = DATA_DIR / split / cls
        count = len(list(cls_dir.glob("*.jpeg"))) if cls_dir.exists() else 0
        summary[split][cls] = count
        print(f"  {split:>5}/{cls:<10}: {count:>5} images")
    total = sum(summary[split].values())
    ratio = summary[split]["PNEUMONIA"] / total * 100 if total > 0 else 0
    print(f"  {split:>5} total: {total:>5} | PNEUMONIA: {ratio:.1f}%")
    print()

# --- Image size statistics ---
print("IMAGE SIZE STATISTICS (sampled from train)")
widths, heights = [], []

for cls in CLASS_NAMES:
    cls_dir = DATA_DIR / "train" / cls
    for img_path in list(cls_dir.glob("*.jpeg"))[:200]:
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)

summary["image_stats"] = {
    "width_min": int(min(widths)), "width_max": int(max(widths)), "width_mean": int(np.mean(widths)),
    "height_min": int(min(heights)), "height_max": int(max(heights)), "height_mean": int(np.mean(heights)),
    "sample_count": len(widths),
}

print(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")

# --- Save class distribution plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, split in enumerate(["train", "val", "test"]):
    counts = summary[split]
    bars = axes[idx].bar(counts.keys(), counts.values(), color=["#2ecc71", "#e74c3c"])
    axes[idx].set_title(f"{split.upper()} Set", fontsize=14)
    axes[idx].set_ylabel("Count")
    for bar, count in zip(bars, counts.values()):
        axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                       str(count), ha="center", fontsize=11)
plt.suptitle("Class Distribution Across Splits", fontsize=16)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {RESULTS_DIR / 'class_distribution.png'}")

# --- Save sample images grid ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for row, cls in enumerate(CLASS_NAMES):
    cls_dir = DATA_DIR / "train" / cls
    images = sorted(cls_dir.glob("*.jpeg"))[:5]
    for col, img_path in enumerate(images):
        img = Image.open(img_path)
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"{cls}\n{img.size[0]}x{img.size[1]}", fontsize=10)
        axes[row, col].axis("off")
plt.suptitle("Sample Chest X-Rays", fontsize=16)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'sample_images.png'}")

# --- Pixel intensity histograms ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, cls in enumerate(CLASS_NAMES):
    cls_dir = DATA_DIR / "train" / cls
    all_pixels = []
    for img_path in list(cls_dir.glob("*.jpeg"))[:50]:
        img = np.array(Image.open(img_path).convert("L").resize((224, 224)))
        all_pixels.extend(img.flatten())
    axes[idx].hist(all_pixels, bins=50, color="steelblue" if idx == 0 else "coral",
                   edgecolor="black", density=True)
    axes[idx].set_title(f"{cls} — Pixel Intensity", fontsize=13)
    axes[idx].set_xlabel("Pixel Value")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pixel_distributions.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'pixel_distributions.png'}")

# --- Save summary JSON ---
with open(RESULTS_DIR / "data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'data_summary.json'}")

print("\nDone! Open the notebook to view all outputs.")
