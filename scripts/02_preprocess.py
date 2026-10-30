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
import cv2
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# These settings control the audit, output folders, image size, and repeatable seed.
DATA_DIR = Path("data/raw/chest_xray")
RESULTS_DIR = Path("experiments/results")
SPLIT_DIR = Path("data/processed/splits")
SPLIT_CSV = SPLIT_DIR / "stratified_split.csv"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 224
SEED = 42


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

 # Check every image and save one traceable row for it in the manifest.
 # Image audit and manifest
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

 # Rebuild the very small validation set, but keep the official test set unchanged.
 # Reproducible train and validation split
def build_stratified_split(rows, val_fraction=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    split_rows = []
    train_val_rows = [r for r in rows if r["split"] in {"train", "val"}]
    test_rows = [r for r in rows if r["split"] == "test"]

    for cls in CLASS_NAMES:
        class_rows = [r for r in train_val_rows if r["class"] == cls]
        indices = np.arange(len(class_rows))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(class_rows) * val_fraction)))
        val_indices = set(indices[:val_count])
        for idx, row in enumerate(class_rows):
            split_name = "val" if idx in val_indices else "train"
            split_rows.append({
                "split": split_name,
                "class": row["class"],
                "path": row["path"],
                "source_split": row["split"],
                "file": row["file"],
                "sha1": row["sha1"],
            })

    for row in test_rows:
        split_rows.append({
            "split": "test",
            "class": row["class"],
            "path": row["path"],
            "source_split": row["split"],
            "file": row["file"],
            "sha1": row["sha1"],
        })

    return split_rows


def save_stratified_split(rows):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = build_stratified_split(rows)
    with open(SPLIT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class", "path", "source_split", "file", "sha1"],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    counts = {
        split: {cls: 0 for cls in CLASS_NAMES}
        for split in SPLITS
    }
    for row in split_rows:
        counts[row["split"]][row["class"]] += 1

    with open(SPLIT_DIR / "stratified_split_summary.json", "w") as f:
        json.dump({
            "split_csv": str(SPLIT_CSV),
            "seed": SEED,
            "validation_fraction_from_train_val_pool": 0.2,
            "official_test_set_preserved": True,
            "counts": counts,
            "note": (
                "The original Kaggle validation split has only 16 images. "
                "This split rebuilds train/val from original train+val while "
                "leaving the official test set untouched."
            ),
        }, f, indent=2)

    print(f"Saved: {SPLIT_CSV}")
    print(f"Saved: {SPLIT_DIR / 'stratified_split_summary.json'}")


def apply_clahe_preview(img):
    # CLAHE improves local contrast. This is shown as a preview; the reported model uses baseline images.
    gray = np.asarray(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray))


def save_preprocessing_preview(rows):
    examples = []
    for cls in CLASS_NAMES:
        examples.extend([r for r in rows if r["split"] == "train" and r["class"] == cls][:3])
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 2, figsize=(8, len(examples) * 3.2))
    axes = np.atleast_2d(axes)
    for idx, row in enumerate(examples):
        img = Image.open(row["path"]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        clahe_img = apply_clahe_preview(img).resize((IMG_SIZE, IMG_SIZE))
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Original: {row['class']}", fontsize=9)
        axes[idx, 1].imshow(clahe_img, cmap="gray")
        axes[idx, 1].set_title("CLAHE preview", fontsize=9)
        for ax in axes[idx]:
            ax.axis("off")
    plt.suptitle("Preprocessing Preview: Original vs CLAHE", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "preprocessing_clahe_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'preprocessing_clahe_preview.png'}")


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
    # This main function runs the complete data audit and saves all plots and JSON/CSV results.
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
    save_stratified_split(rows)
    save_plots(rows, summary)
    save_preprocessing_preview(rows)

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
    print(f"  Stratified split CSV:   {SPLIT_CSV}")
    print(f"  Class distribution plot:{RESULTS_DIR / 'class_distribution.png'}")
    print(f"  Sample image grid:      {RESULTS_DIR / 'sample_images.png'}")
    print(f"  Pixel distribution plot:{RESULTS_DIR / 'pixel_distributions.png'}")
    print(f"  CLAHE preview:          {RESULTS_DIR / 'preprocessing_clahe_preview.png'}")
    if bad_images:
        print(f"  Bad image log:          {RESULTS_DIR / 'bad_images.json'}")

    print("\nDone. Run scripts/03_train_classifier.py next.")


if __name__ == "__main__":
    main()
