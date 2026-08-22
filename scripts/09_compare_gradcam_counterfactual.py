"""
Step 9: Compare Grad-CAM and counterfactual explanations side-by-side.

Usage:
    python scripts/09_compare_gradcam_counterfactual.py
    python scripts/09_compare_gradcam_counterfactual.py --max-images 8

What it does:
    - This is the key comparison step for the dissertation
    - For each test image, shows: Original -> Grad-CAM -> Standard CF -> Guided CF
    - Answers the central research question: "Does counterfactual explanation provide
      a clearer picture of model reasoning than heatmaps alone?"
    - Grad-CAM shows WHERE the model looked (attention)
    - Counterfactual shows WHAT would need to change (decision boundary)
    - Guided CF combines both: focused changes on attention regions
    - Generates overlap analysis, agreement metrics, and publication-ready panels
    - Produces a master comparison grid and per-image detailed panels
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

GRADCAM_DIR = Path("outputs/gradcam")
CF_DIR = Path("outputs/counterfactuals")
COMP_DIR = Path("outputs/comparisons")
GUIDED_CF_DIR = Path("outputs/guided_counterfactuals")
QUALITY_DIR = Path("outputs/explanation_quality")
OUTPUT_DIR = Path("outputs/gradcam_vs_counterfactual")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_matching_files():
    gradcam_meta = load_json(GRADCAM_DIR / "gradcam_metadata.json")
    cf_meta = load_json(CF_DIR / "counterfactual_metadata.json")
    guided_meta = load_json(GUIDED_CF_DIR / "guided_cf_metadata.json")

    if not gradcam_meta or not cf_meta:
        raise SystemExit(
            "Need both Grad-CAM and counterfactual outputs. "
            "Run steps 05 and 06 first."
        )

    gc_files = {e["file"]: e for e in gradcam_meta.get("selected", [])}
    cf_files = {e["file"]: e for e in cf_meta.get("selected", [])}
    guided_files = {}
    if guided_meta:
        guided_files = {e["file"]: e for e in guided_meta.get("selected", [])}

    common = set(gc_files.keys()) & set(cf_files.keys())
    return sorted(common), gc_files, cf_files, guided_files


def find_image(directory, file_stem, suffixes):
    for suffix in suffixes:
        candidates = list(directory.glob(f"*{file_stem}*{suffix}"))
        if candidates:
            return candidates[0]
    return None


def load_resized(path):
    if path is None or not path.exists():
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return np.asarray(
        Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    )


def compute_overlap(gradcam_path, diff_path):
    if gradcam_path is None or diff_path is None:
        return 0.0
    if not gradcam_path.exists() or not diff_path.exists():
        return 0.0

    gc = np.asarray(Image.open(gradcam_path).convert("L").resize((IMG_SIZE, IMG_SIZE)))
    gc_norm = gc.astype(np.float32) / max(gc.max(), 1)
    gc_binary = (gc_norm > 0.3).astype(np.float32)

    diff = np.asarray(Image.open(diff_path).convert("L").resize((IMG_SIZE, IMG_SIZE)))
    diff_norm = diff.astype(np.float32) / max(diff.max(), 1)
    diff_binary = (diff_norm > 0.15).astype(np.float32)

    intersection = (gc_binary * diff_binary).sum()
    union = np.clip(gc_binary + diff_binary, 0, 1).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def save_detailed_panel(
    filename, gc_entry, cf_entry, guided_entry,
    overlap_score, output_dir, idx,
):
    file_stem = Path(filename).stem
    true_label = gc_entry.get("true_label", "?")
    pred_label = gc_entry.get("pred_label", "?")
    confidence = gc_entry.get("confidence", 0)
    correct = gc_entry.get("correct", False)

    original_path = gc_entry.get("path", "")
    original = load_resized(Path(original_path)) if original_path else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    gc_overlay = load_resized(find_image(GRADCAM_DIR, file_stem, ["_overlay.png"]))
    gc_heatmap = load_resized(find_image(GRADCAM_DIR, file_stem, ["_heatmap.png"]))

    cf_image = load_resized(find_image(CF_DIR, file_stem, ["_counterfactual.png"]))
    cf_diff = load_resized(find_image(CF_DIR, file_stem, ["_diff_map.png"]))

    has_guided = bool(guided_entry)
    guided_cf = load_resized(
        find_image(GUIDED_CF_DIR, file_stem, ["_guided_cf.png"])
    ) if has_guided else None
    guided_diff = load_resized(
        find_image(GUIDED_CF_DIR, file_stem, ["_guided_diff.png"])
    ) if has_guided else None

    n_cols = 7 if has_guided else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.8, 5.8))

    axes[0].imshow(original)
    axes[0].set_title(
        "1. Original X-ray\n"
        f"True label: {true_label}",
        fontsize=10,
    )

    axes[1].imshow(gc_overlay)
    axes[1].set_title(
        "2. Model focus on X-ray\n"
        "Bright colors = looked here more",
        fontsize=10,
    )

    axes[2].imshow(gc_heatmap)
    axes[2].set_title(
        "3. Focus map alone\n"
        "Bright = stronger model focus",
        fontsize=10,
    )

    cf_label = cf_entry.get("cf_pred_label", "?")
    cf_conf = cf_entry.get("cf_confidence", 0)
    cf_success = cf_entry.get("cf_success", False)
    cf_status = "flipped" if cf_success else "partial"
    axes[3].imshow(cf_image)
    axes[3].set_title(
        "4. Changed X-ray\n"
        "Synthetic edited image\n"
        f"New prediction: {cf_label} ({cf_conf:.3f})",
        fontsize=10,
    )

    axes[4].imshow(cf_diff)
    axes[4].set_title(
        "5. Needed changes\n"
        "Colors show what had to change",
        fontsize=10,
    )

    if has_guided and guided_cf is not None:
        g_label = guided_entry.get("guided_cf_pred_label", "?")
        g_conf = guided_entry.get("guided_cf_confidence", 0)
        g_success = guided_entry.get("guided_cf_success", False)
        g_status = "flipped" if g_success else "partial"
        l1_red = guided_entry.get("l1_reduction_pct", 0)

        axes[5].imshow(guided_cf)
        axes[5].set_title(
            "6. Focused changed X-ray\n"
            "Focused synthetic edit\n"
            f"New prediction: {g_label} ({g_conf:.3f})",
            fontsize=10,
        )

        axes[6].imshow(guided_diff)
        axes[6].set_title(
            "7. Focused needed changes\n"
            f"Usually smaller edits\nChange reduction: {l1_red:.0f}%",
            fontsize=10,
        )

    for ax in axes:
        ax.axis("off")

    status = "correct" if correct else "MISCLASSIFIED"
    fig.suptitle(
        f"Explanation comparison for {filename}  |  original prediction: {pred_label} ({confidence:.3f})  |  "
        f"focus/change agreement: {overlap_score:.2f}",
        fontsize=12, y=0.99,
    )
    fig.text(
        0.5,
        0.035,
        "How to read this: the focus map shows where the model looked. The changed images show what pixels had "
        "to change to flip the model answer. If the colored regions match, the explanation is more consistent.",
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.90])
    panel_path = output_dir / f"{idx:02d}_{file_stem}_comparison.png"
    plt.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def save_master_grid(results, output_dir):
    if not results:
        return

    has_guided = any(r.get("has_guided", False) for r in results)
    n_cols = 5 if has_guided else 4
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = np.atleast_2d(axes)

    headers = ["Original", "Model Focus", "Normal Changes", "Focused Changes", "Agreement"]
    if not has_guided:
        headers = ["Original", "Model Focus", "Changed X-ray", "Needed Changes"]

    for col, header in enumerate(headers[:n_cols]):
        axes[0, col].set_title(header, fontsize=10, fontweight="bold")

    for row_idx, r in enumerate(results):
        file_stem = Path(r["file"]).stem

        original_path = r.get("path", "")
        original = load_resized(Path(original_path)) if original_path else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        gc_overlay = load_resized(find_image(GRADCAM_DIR, file_stem, ["_overlay.png"]))
        cf_diff = load_resized(find_image(CF_DIR, file_stem, ["_diff_map.png"]))

        axes[row_idx, 0].imshow(original)
        status = "OK" if r.get("correct", False) else "MISS"
        axes[row_idx, 0].set_ylabel(
            f"{status}\n{r.get('true_label', '?')}",
            fontsize=8, rotation=0, labelpad=50, va="center",
        )

        axes[row_idx, 1].imshow(gc_overlay)

        if has_guided:
            axes[row_idx, 2].imshow(cf_diff)

            guided_diff = load_resized(
                find_image(GUIDED_CF_DIR, file_stem, ["_guided_diff.png"])
            )
            axes[row_idx, 3].imshow(guided_diff)

            overlap = r.get("overlap_score", 0)
            color = "#22c55e" if overlap > 0.3 else "#f59e0b" if overlap > 0.15 else "#ef4444"
            axes[row_idx, 4].barh([0], [overlap], color=color, height=0.5)
            axes[row_idx, 4].set_xlim(0, 1)
            axes[row_idx, 4].text(overlap + 0.02, 0, f"{overlap:.2f}", va="center", fontsize=9)
            axes[row_idx, 4].set_yticks([])
        else:
            cf_image = load_resized(find_image(CF_DIR, file_stem, ["_counterfactual.png"]))
            axes[row_idx, 2].imshow(cf_image)
            axes[row_idx, 3].imshow(cf_diff)

        for ax in axes[row_idx]:
            ax.axis("off")

    title = "Where the Model Looked vs What Had to Change"
    if has_guided:
        title = "Model Focus vs Normal Changes vs Focused Changes"
    plt.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    plt.savefig(output_dir / "master_comparison_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overlap_chart(results, output_dir):
    if not results:
        return

    files = [r["file"][:20] for r in results]
    overlaps = [r.get("overlap_score", 0) for r in results]
    colors = ["#22c55e" if o > 0.3 else "#f59e0b" if o > 0.15 else "#ef4444"
              for o in overlaps]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(files)), overlaps, color=colors)
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Agreement score")
    ax.set_title(
        "Do the focus colors and change colors appear in the same places?\n"
        "High = model focus and needed changes agree. "
        "Low = the model looked somewhere different from the pixels that changed.",
    )
    ax.axhline(y=0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, max(max(overlaps) + 0.1, 0.5))

    for bar, val in zip(bars, overlaps):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "overlap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common_files, gc_files, cf_files, guided_files = find_matching_files()
    print(f"Found {len(common_files)} images with both Grad-CAM and CF outputs")
    if guided_files:
        guided_common = set(common_files) & set(guided_files.keys())
        print(f"  of which {len(guided_common)} also have guided CF outputs")

    results = []
    for idx, filename in enumerate(common_files):
        gc_entry = gc_files[filename]
        cf_entry = cf_files[filename]
        guided_entry = guided_files.get(filename)

        file_stem = Path(filename).stem
        gc_heatmap_path = find_image(GRADCAM_DIR, file_stem, ["_heatmap.png"])
        cf_diff_path = find_image(CF_DIR, file_stem, ["_diff_map.png"])
        overlap = compute_overlap(gc_heatmap_path, cf_diff_path)

        panel_path = save_detailed_panel(
            filename, gc_entry, cf_entry, guided_entry,
            overlap, OUTPUT_DIR, idx,
        )

        result = {
            "file": filename,
            "path": gc_entry.get("path", ""),
            "true_label": gc_entry.get("true_label", "?"),
            "pred_label": gc_entry.get("pred_label", "?"),
            "correct": gc_entry.get("correct", False),
            "confidence": gc_entry.get("confidence", 0),
            "cf_success": cf_entry.get("cf_success", False),
            "cf_pred_label": cf_entry.get("cf_pred_label", "?"),
            "overlap_score": round(overlap, 4),
            "has_guided": guided_entry is not None,
            "panel_path": str(panel_path),
        }
        if guided_entry:
            result["guided_cf_success"] = guided_entry.get("guided_cf_success", False)
            result["l1_reduction_pct"] = guided_entry.get("l1_reduction_pct", 0)

        results.append(result)
        print(f"  [{idx + 1}/{len(common_files)}] {filename} — overlap={overlap:.2f}")

    save_master_grid(results, OUTPUT_DIR)
    save_overlap_chart(results, OUTPUT_DIR)

    quality_data = load_json(QUALITY_DIR / "explanation_quality.json")

    summary = {
        "num_compared": len(results),
        "mean_overlap": round(float(np.mean(
            [r["overlap_score"] for r in results]
        )), 4) if results else 0,
        "high_overlap_count": sum(1 for r in results if r["overlap_score"] > 0.3),
        "cf_success_rate": round(
            sum(1 for r in results if r["cf_success"]) / max(len(results), 1), 4
        ),
        "per_image": [{k: v for k, v in r.items()
                       if k != "panel_path"} for r in results],
    }
    if quality_data:
        summary["gradcam_faithfulness"] = quality_data.get("gradcam", {}).get(
            "mean_faithfulness", 0
        )
        summary["cf_mean_ssim"] = quality_data.get("standard_counterfactual", {}).get(
            "mean_ssim", 0
        )
        if quality_data.get("guided_counterfactual", {}).get("num_evaluated", 0) > 0:
            summary["guided_cf_mean_ssim"] = quality_data["guided_counterfactual"]["mean_ssim"]

    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison complete.")
    print(f"  Mean Grad-CAM/CF overlap: {summary['mean_overlap']:.4f}")
    print(f"  High overlap (>0.3): {summary['high_overlap_count']}/{len(results)}")
    print("\nStep 9 outputs")
    print(f"  Output directory:       {OUTPUT_DIR}")
    print(f"  Comparison summary JSON:{OUTPUT_DIR / 'comparison_summary.json'}")
    print(f"  Master grid:            {OUTPUT_DIR / 'master_comparison_grid.png'}")
    print(f"  Overlap chart:          {OUTPUT_DIR / 'overlap_analysis.png'}")
    print(f"  Per-image comparisons:  {OUTPUT_DIR / '*_comparison.png'}")
    print(f"\nDone! Run 10_generate_report.py next.")


if __name__ == "__main__":
    main()
