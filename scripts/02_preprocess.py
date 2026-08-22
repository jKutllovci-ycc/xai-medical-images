"""
Step 2: Explore the dataset and write training recommendations.

Usage:
    python3 scripts/02_preprocess.py

What it does:
    - Counts images per class per split
    - Computes image size and pixel statistics
    - Builds a manifest CSV for auditability
    - Estimates class weights for imbalanced training
    - Saves plots and JSON summaries under experiments/results/
"""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib
import numpy as np
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path("data/raw/chest_xray")
RESULTS_DIR = Path("experiments/results")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 224


def iter_images(split=None, class_name=None):
    splits = [split] if split else SPLITS
    classes = [class_name] if class_name else CLASS_NAMES
    for split_name in splits:
        for cls in classes:
            cls_dir = DATA_DIR / split_name / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.jpeg")):
                yield split_name, cls, img_path


def image_sha1(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_open(path):
    try:
        img = Image.open(path)
        img.verify()
        return Image.open(path).convert("L")
    except Exception as exc:
        return None, str(exc)


def compute_class_weights(counts):
    total = sum(counts.get(cls, 0) for cls in CLASS_NAMES)
    num_classes = len(CLASS_NAMES)
    return {
        cls: float(total / (num_classes * max(counts.get(cls, 0), 1)))
        for cls in CLASS_NAMES
    }


def build_manifest():
    rows = []
    bad_images = []
    hashes = Counter()

    for split, cls, img_path in iter_images():
        opened = safe_open(img_path)
        if isinstance(opened, tuple):
            bad_images.append({"path": str(img_path), "error": opened[1]})
            continue

        img = opened
        digest = image_sha1(img_path)
        hashes[digest] += 1
        arr = np.asarray(ImageOps.autocontrast(img).resize((IMG_SIZE, IMG_SIZE)))
        rows.append({
            "split": split,
            "class": cls,
            "path": str(img_path),
            "file": img_path.name,
            "width": img.width,
            "height": img.height,
            "aspect_ratio": round(img.width / max(img.height, 1), 4),
            "mean_intensity": round(float(arr.mean()), 4),
            "std_intensity": round(float(arr.std()), 4),
            "sha1": digest,
        })

    duplicate_hashes = {digest for digest, count in hashes.items() if count > 1}
    for row in rows:
        row["duplicate_content"] = row["sha1"] in duplicate_hashes

    return rows, bad_images, len(duplicate_hashes)


def summarize(rows, bad_images, duplicate_hash_count):
    summary = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        counts = Counter(row["class"] for row in split_rows)
        summary[split] = {cls: int(counts.get(cls, 0)) for cls in CLASS_NAMES}
        total = sum(summary[split].values())
        summary[split]["total"] = total
        summary[split]["pneumonia_percent"] = (
            round(summary[split]["PNEUMONIA"] / total * 100, 2) if total else 0.0
        )

    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    means = [row["mean_intensity"] for row in rows]
    stds = [row["std_intensity"] for row in rows]
    pixel_stats_by_split_class = {}
    for split in SPLITS:
        pixel_stats_by_split_class[split] = {}
        for cls in CLASS_NAMES:
            class_rows = [row for row in rows if row["split"] == split and row["class"] == cls]
            class_means = [row["mean_intensity"] for row in class_rows]
            class_stds = [row["std_intensity"] for row in class_rows]
            pixel_stats_by_split_class[split][cls] = {
                "image_count": len(class_rows),
                "mean_intensity_mean": round(float(np.mean(class_means)), 4) if class_means else 0.0,
                "mean_intensity_std": round(float(np.std(class_means)), 4) if class_means else 0.0,
                "std_intensity_mean": round(float(np.mean(class_stds)), 4) if class_stds else 0.0,
                "std_intensity_std": round(float(np.std(class_stds)), 4) if class_stds else 0.0,
            }

    summary["image_stats"] = {
        "width_min": int(min(widths)) if widths else 0,
        "width_max": int(max(widths)) if widths else 0,
        "width_mean": round(float(np.mean(widths)), 2) if widths else 0.0,
        "height_min": int(min(heights)) if heights else 0,
        "height_max": int(max(heights)) if heights else 0,
        "height_mean": round(float(np.mean(heights)), 2) if heights else 0.0,
        "mean_intensity": round(float(np.mean(means)), 4) if means else 0.0,
        "std_intensity": round(float(np.mean(stds)), 4) if stds else 0.0,
        "sample_count": len(rows),
    }
    summary["pixel_intensity_by_split_class"] = pixel_stats_by_split_class
    summary["data_quality"] = {
        "bad_image_count": len(bad_images),
        "duplicate_content_hash_count": duplicate_hash_count,
        "duplicate_file_count": int(sum(1 for row in rows if row["duplicate_content"])),
    }
    summary["training_recommendations"] = {
        "train_class_weights": compute_class_weights(summary["train"]),
        "avoid_horizontal_flip": True,
        "reason": "Chest X-rays include laterality markers and asymmetric anatomy; flips can encourage shortcuts.",
    }
    if summary["val"]["total"] < 100:
        summary["training_recommendations"]["validation_warning"] = (
            f"Validation split has only {summary['val']['total']} images; prefer balanced accuracy "
            "and confirm every checkpoint on the held-out test set."
        )

    return summary


def save_manifest(rows):
    if not rows:
        return
    path = RESULTS_DIR / "dataset_manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_plots(rows, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, split in enumerate(SPLITS):
        counts = [summary[split][cls] for cls in CLASS_NAMES]
        bars = axes[idx].bar(CLASS_NAMES, counts, color=["#3b82f6", "#ef4444"])
        axes[idx].set_title(f"{split.upper()} Set", fontsize=13)
        axes[idx].set_ylabel("Count")
        for bar, count in zip(bars, counts):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           str(count), ha="center", va="bottom", fontsize=10)
    plt.suptitle("Class Distribution Across Splits", fontsize=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'class_distribution.png'}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for row_idx, cls in enumerate(CLASS_NAMES):
        images = [row for row in rows if row["split"] == "train" and row["class"] == cls][:5]
        for col_idx, row in enumerate(images):
            img = Image.open(row["path"]).convert("L")
            axes[row_idx, col_idx].imshow(img, cmap="gray")
            axes[row_idx, col_idx].set_title(
                f"{cls}\n{row['width']}x{row['height']}", fontsize=10
            )
            axes[row_idx, col_idx].axis("off")
    plt.suptitle("Sample Chest X-Rays", fontsize=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'sample_images.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, cls in enumerate(CLASS_NAMES):
        class_rows = [row for row in rows if row["split"] == "train" and row["class"] == cls]
        pixels = []
        for row in class_rows[:80]:
            img = Image.open(row["path"]).convert("L").resize((IMG_SIZE, IMG_SIZE))
            pixels.append(np.asarray(img).reshape(-1))
        if pixels:
            axes[idx].hist(np.concatenate(pixels), bins=50,
                           color="#3b82f6" if idx == 0 else "#ef4444",
                           edgecolor="black", density=True)
        axes[idx].set_title(f"{cls} Pixel Intensity", fontsize=13)
        axes[idx].set_xlabel("Pixel Value")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pixel_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'pixel_distributions.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DATASET AUDIT")
    print("=" * 50)
    if not DATA_DIR.exists():
        raise SystemExit(f"Missing dataset directory: {DATA_DIR}. Run scripts/01_download_data.py first.")

    rows, bad_images, duplicate_hash_count = build_manifest()
    if not rows:
        raise SystemExit(f"No readable JPEG images found under {DATA_DIR}.")

    summary = summarize(rows, bad_images, duplicate_hash_count)
    for split in SPLITS:
        print(
            f"  {split:>5}: NORMAL={summary[split]['NORMAL']:>4} | "
            f"PNEUMONIA={summary[split]['PNEUMONIA']:>4} | "
            f"total={summary[split]['total']:>4} | "
            f"pneumonia={summary[split]['pneumonia_percent']:.1f}%"
        )
    print(f"  duplicate content hashes: {duplicate_hash_count}")
    print(f"  unreadable images: {len(bad_images)}")

    save_manifest(rows)
    save_plots(rows, summary)

    with open(RESULTS_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'data_summary.json'}")

    if bad_images:
        with open(RESULTS_DIR / "bad_images.json", "w") as f:
            json.dump(bad_images, f, indent=2)
        print(f"Saved: {RESULTS_DIR / 'bad_images.json'}")

    print("\nStep 2 outputs")
    print(f"  Output directory:       {RESULTS_DIR}")
    print(f"  Dataset manifest:       {RESULTS_DIR / 'dataset_manifest.csv'}")
    print(f"  Dataset summary:        {RESULTS_DIR / 'data_summary.json'}")
    print(f"  Class distribution plot:{RESULTS_DIR / 'class_distribution.png'}")
    print(f"  Sample image grid:      {RESULTS_DIR / 'sample_images.png'}")
    print(f"  Pixel distribution plot:{RESULTS_DIR / 'pixel_distributions.png'}")
    if bad_images:
        print(f"  Bad image log:          {RESULTS_DIR / 'bad_images.json'}")

    print("\nDone. Run scripts/03_train_classifier.py next.")


if __name__ == "__main__":
    main()
