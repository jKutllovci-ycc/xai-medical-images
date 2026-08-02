"""
Step 5: Generate sharper Grad-CAM explanations.

Usage:
    python3 scripts/05_gradcam.py
    python3 scripts/05_gradcam.py --method xgradcam --target-class predicted

What it does:
    - Loads the trained classifier
    - Scores the full test set and chooses informative examples
    - Generates class-targeted Grad-CAM maps
    - Suppresses obvious non-anatomy/background regions
    - Saves individual overlays, masks, heatmaps, a summary grid, and metadata
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DROPOUT = 0.3
NUM_CLASSES = 2

DATA_DIR = Path("data/raw/chest_xray")
CHECKPOINT_PATH = Path("experiments/checkpoints/best_classifier.pt")
OUTPUT_DIR = Path("outputs/gradcam")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
}


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples, self.labels = [], []
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob("*.jpeg")):
                    self.samples.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx], str(self.samples[idx])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model(device):
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"Missing checkpoint: {CHECKPOINT_PATH}. Run scripts/03_train_classifier.py first."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(
        f"Loaded checkpoint epoch={checkpoint.get('epoch', '?')} "
        f"val_loss={checkpoint.get('val_loss', float('nan')):.4f}"
    )
    return model


def score_test_set(model, dataset, device):
    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers)
    rows = []
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Scoring test set"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for path, true_idx, pred_idx, prob in zip(paths, labels.numpy(), preds, probs):
                confidence = float(prob[pred_idx])
                true_prob = float(prob[true_idx])
                rows.append({
                    "path": path,
                    "file": Path(path).name,
                    "true_idx": int(true_idx),
                    "true_label": CLASS_NAMES[int(true_idx)],
                    "pred_idx": int(pred_idx),
                    "pred_label": CLASS_NAMES[int(pred_idx)],
                    "confidence": confidence,
                    "normal_prob": float(prob[0]),
                    "pneumonia_prob": float(prob[1]),
                    "true_class_prob": true_prob,
                    "correct": bool(pred_idx == true_idx),
                })
    return rows


def select_examples(rows, max_per_class):
    selected = []
    seen = set()

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_rows = [r for r in rows if r["true_idx"] == class_idx]
        correct = sorted(
            [r for r in class_rows if r["correct"]],
            key=lambda r: r["confidence"],
            reverse=True,
        )
        hard = sorted(
            class_rows,
            key=lambda r: (r["correct"], r["true_class_prob"]),
        )

        quota_easy = max(1, max_per_class // 2)
        candidates = correct[:quota_easy] + hard[:max_per_class]
        count = 0
        for row in candidates:
            if row["path"] in seen:
                continue
            selected.append(row)
            seen.add(row["path"])
            count += 1
            if count >= max_per_class:
                break

    return selected


def load_display_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


def make_lung_field_mask(rgb):
    """Approximate lung fields from dark aerated regions, then expand to cover opacities."""
    gray = (rgb.mean(axis=2) * 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, body = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        body = np.where(labels == largest, 255, 0).astype(np.uint8)

    margin = int(IMG_SIZE * 0.04)
    body[:, :margin] = 0
    body[:, -margin:] = 0
    body[:margin, :] = 0
    body[-margin:, :] = 0

    body_pixels = gray[body > 0]
    if body_pixels.size < 10:
        return body.astype(np.float32) / 255.0

    dark_cutoff = np.percentile(body_pixels, 58)
    very_dark = np.percentile(body_pixels, 3)
    candidate = ((gray <= dark_cutoff) & (gray >= very_dark) & (body > 0)).astype(np.uint8) * 255

    roi = np.zeros_like(candidate)
    roi[int(IMG_SIZE * 0.12):int(IMG_SIZE * 0.88),
        int(IMG_SIZE * 0.08):int(IMG_SIZE * 0.92)] = 255
    candidate = cv2.bitwise_and(candidate, roi)

    center_band = np.zeros_like(candidate)
    center_band[:, int(IMG_SIZE * 0.42):int(IMG_SIZE * 0.58)] = 255
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(center_band))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(candidate)
    min_area = int(0.012 * IMG_SIZE * IMG_SIZE)
    max_area = int(0.35 * IMG_SIZE * IMG_SIZE)
    plausible = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h < IMG_SIZE * 0.18 or w < IMG_SIZE * 0.08:
            continue
        cx = x + w / 2
        if IMG_SIZE * 0.12 <= cx <= IMG_SIZE * 0.88:
            plausible.append((area, contour))

    for _, contour in sorted(plausible, key=lambda item: item[0], reverse=True)[:4]:
        hull = cv2.convexHull(contour)
        cv2.drawContours(kept, [hull], -1, 255, thickness=-1)

    if not kept.any():
        kept = candidate

    kept = cv2.dilate(kept, np.ones((19, 19), np.uint8), iterations=1)
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    kept = cv2.bitwise_and(kept, body)
    return kept.astype(np.float32) / 255.0


def percentile_normalize(cam, mask, low=55, high=99.3):
    masked_values = cam[mask > 0.1]
    if masked_values.size < 10:
        masked_values = cam.reshape(-1)

    lo = np.percentile(masked_values, low)
    hi = np.percentile(masked_values, high)
    if hi <= lo:
        hi = float(masked_values.max())
        lo = float(masked_values.min())
    if hi <= lo:
        return np.zeros_like(cam, dtype=np.float32)

    cam = np.clip((cam - lo) / (hi - lo), 0, 1)
    cam = cam * mask
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=1.0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def make_overlay(rgb, cam, threshold=0.18, alpha=0.65):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha_map = (np.clip((cam - threshold) / (1 - threshold), 0, 1) ** 0.7)
    alpha_map = (alpha * alpha_map)[..., None]
    overlay = rgb * (1 - alpha_map) + heatmap * alpha_map
    return np.uint8(np.clip(overlay, 0, 1) * 255)


def save_single_result(row, rgb, cam, mask, overlay, output_dir, idx):
    true_label = row["true_label"].lower()
    stem = f"{idx:02d}_{true_label}_{Path(row['path']).stem}"

    plt.imsave(output_dir / f"{stem}_heatmap.png", cam, cmap="turbo", vmin=0, vmax=1)
    plt.imsave(output_dir / f"{stem}_mask.png", mask, cmap="gray", vmin=0, vmax=1)
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Original\nTrue: {row['true_label']}", fontsize=10)
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Lung-field Mask", fontsize=10)
    axes[2].imshow(cam, cmap="turbo", vmin=0, vmax=1)
    axes[2].set_title("Sharpened CAM", fontsize=10)
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Pred: {row['pred_label']}\nconf={row['confidence']:.3f}",
        fontsize=10,
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(Path(row["path"]).name, fontsize=11)
    plt.tight_layout()
    panel_path = output_dir / f"{stem}_panel.png"
    plt.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def save_comparison_result(row, rgb, legacy_overlay, improved_overlay, mask,
                           output_dir, idx):
    true_label = row["true_label"].lower()
    stem = f"{idx:02d}_{true_label}_{Path(row['path']).stem}"
    comparison_path = output_dir / f"{stem}_legacy_vs_improved.png"

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title(
        f"Original\nTrue: {row['true_label']} | Pred: {row['pred_label']}",
        fontsize=10,
    )
    axes[1].imshow(legacy_overlay)
    axes[1].set_title(
        "Before\nVanilla Grad-CAM\npredicted class, no mask",
        fontsize=10,
    )
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Added\nlung-field mask", fontsize=10)
    axes[3].imshow(improved_overlay)
    axes[3].set_title(
        "Now\nGrad-CAM++\npneumonia target + sharpened mask",
        fontsize=10,
    )

    for ax in axes:
        ax.axis("off")

    status = "correct" if row["correct"] else "misclassified"
    fig.suptitle(
        f"{Path(row['path']).name} | {status} | confidence={row['confidence']:.3f}",
        fontsize=11,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return comparison_path


def cam_quality(cam, mask):
    total = float(cam.sum())
    if total <= 0:
        return {"cam_in_mask_fraction": 0.0, "cam_peak": 0.0, "cam_area_50": 0.0}
    in_mask = float((cam * mask).sum() / total)
    return {
        "cam_in_mask_fraction": in_mask,
        "cam_peak": float(cam.max()),
        "cam_area_50": float((cam >= 0.5).mean()),
    }


def save_summary_grid(results, output_dir):
    if not results:
        return
    cols = 3
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.6))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.reshape(-1):
        ax.axis("off")

    for ax, result in zip(axes.reshape(-1), results):
        ax.imshow(result["overlay"])
        status = "OK" if result["correct"] else "MISS"
        ax.set_title(
            f"{status} {result['true_label']} -> {result['pred_label']} "
            f"({result['confidence']:.2f})\n{result['file']}",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_summary_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(results, output_dir):
    if not results or "legacy_overlay" not in results[0]:
        return

    rows = len(results)
    fig, axes = plt.subplots(rows, 2, figsize=(9, rows * 3.8))
    axes = np.atleast_2d(axes)

    for row_idx, result in enumerate(results):
        status = "OK" if result["correct"] else "MISS"
        axes[row_idx, 0].imshow(result["legacy_overlay"])
        axes[row_idx, 0].set_title(
            f"Before: vanilla Grad-CAM\n{status} {result['true_label']} -> "
            f"{result['pred_label']} ({result['confidence']:.2f})",
            fontsize=8,
        )
        axes[row_idx, 1].imshow(result["overlay"])
        axes[row_idx, 1].set_title(
            f"Now: {result['target_label']}-targeted Grad-CAM++\n"
            "lung mask + percentile sharpening",
            fontsize=8,
        )
        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_before_after_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_metadata(results, all_rows, output_dir, args):
    metadata = {
        "method": args.method,
        "target_class": args.target_class,
        "target_layer": args.target_layer,
        "aug_smooth": args.aug_smooth,
        "eigen_smooth": args.eigen_smooth,
        "max_per_class": args.max_per_class,
        "selected_count": len(results),
        "all_test_count": len(all_rows),
        "selected": [
            {k: v for k, v in result.items()
             if k not in {"overlay", "legacy_overlay"}}
            for result in results
        ],
    }
    with open(output_dir / "gradcam_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    fieldnames = [
        "file", "path", "true_label", "pred_label", "correct", "confidence",
        "normal_prob", "pneumonia_prob", "target_label", "cam_in_mask_fraction",
        "cam_peak", "cam_area_50", "panel_path", "comparison_path",
    ]
    with open(output_dir / "gradcam_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})

    with open(output_dir / "test_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), default="gradcam++",
                        help="CAM method. GradCAM++ is usually less diffuse than vanilla Grad-CAM.")
    parser.add_argument("--max-per-class", type=int, default=6,
                        help="Number of selected test examples per true class.")
    parser.add_argument("--target-class", choices=["predicted", "true", "pneumonia"],
                        default="pneumonia",
                        help="Class to explain for each image.")
    parser.add_argument("--target-layer", choices=["layer3", "layer4", "both"],
                        default="layer4",
                        help="ResNet block to explain. layer3 is higher-resolution; layer4 is more semantic.")
    parser.add_argument("--aug-smooth", action="store_true",
                        help="Average CAM over test-time augmentations. Slower, often smoother.")
    parser.add_argument("--eigen-smooth", action="store_true",
                        help="Use PCA denoising from pytorch-grad-cam.")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep old Grad-CAM files in the output directory.")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip before/after panels comparing legacy and improved Grad-CAM.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        for path in OUTPUT_DIR.glob("*"):
            if path.is_file():
                path.unlink()

    device = get_device()
    print(f"Using device: {device}")
    transform = build_eval_transform()
    dataset = ChestXRayDataset(DATA_DIR, "test", transform)
    if len(dataset) == 0:
        raise SystemExit(
            f"No test images found under {DATA_DIR}. Run scripts/01_download_data.py first."
        )

    model = build_model(device)
    all_rows = score_test_set(model, dataset, device)
    selected_rows = select_examples(all_rows, args.max_per_class)
    print(f"Selected {len(selected_rows)} informative examples from {len(all_rows)} test images")

    if args.target_layer == "layer3":
        target_layers = [model.layer3[-1]]
    elif args.target_layer == "layer4":
        target_layers = [model.layer4[-1]]
    else:
        target_layers = [model.layer3[-1], model.layer4[-1]]
    cam_cls = METHODS[args.method]
    cam = cam_cls(model=model, target_layers=target_layers)
    legacy_cam = None
    if not args.no_comparison:
        legacy_cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    results = []
    for idx, row in enumerate(tqdm(selected_rows, desc="Generating CAMs")):
        rgb = load_display_image(row["path"])
        mask = make_lung_field_mask(rgb)
        input_tensor = transform(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)

        if args.target_class == "true":
            target_idx = row["true_idx"]
        elif args.target_class == "pneumonia":
            target_idx = CLASS_NAMES.index("PNEUMONIA")
        else:
            target_idx = row["pred_idx"]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_idx)],
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )[0]
        sharpened_cam = percentile_normalize(grayscale_cam, mask)
        overlay = make_overlay(rgb, sharpened_cam)
        panel_path = save_single_result(row, rgb, sharpened_cam, mask, overlay, OUTPUT_DIR, idx)
        comparison_path = ""
        legacy_overlay = None
        if legacy_cam is not None:
            legacy_cam_map = legacy_cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(row["pred_idx"])],
                aug_smooth=False,
                eigen_smooth=False,
            )[0]
            legacy_overlay = show_cam_on_image(
                rgb.astype(np.float32),
                legacy_cam_map,
                use_rgb=True,
            )
            comparison_path = save_comparison_result(
                row, rgb, legacy_overlay, overlay, mask, OUTPUT_DIR, idx
            )

        quality = cam_quality(sharpened_cam, mask)
        result = {
            **row,
            **quality,
            "target_idx": int(target_idx),
            "target_label": CLASS_NAMES[int(target_idx)],
            "panel_path": str(panel_path),
            "comparison_path": str(comparison_path),
            "overlay": overlay,
        }
        if legacy_overlay is not None:
            result["legacy_overlay"] = legacy_overlay
        results.append(result)

    save_summary_grid(results, OUTPUT_DIR)
    save_comparison_grid(results, OUTPUT_DIR)
    write_metadata(results, all_rows, OUTPUT_DIR, args)

    low_focus = [r for r in results if r["cam_in_mask_fraction"] < 0.75]
    misses = [r for r in results if not r["correct"]]

    print(f"\nSaved Grad-CAM outputs to {OUTPUT_DIR}/")
    print(f"  Summary grid: {OUTPUT_DIR / 'gradcam_summary_grid.png'}")
    if not args.no_comparison:
        print(f"  Before/after: {OUTPUT_DIR / 'gradcam_before_after_grid.png'}")
    print(f"  Metadata:     {OUTPUT_DIR / 'gradcam_metadata.json'}")
    print(f"  Predictions:  {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"  Selected misses: {len(misses)}")
    print(f"  Low in-mask focus (<0.75): {len(low_focus)}")
    if low_focus:
        print("  Review these low-focus examples:")
        for row in low_focus[:5]:
            print(f"    - {row['file']} ({row['true_label']} -> {row['pred_label']}, "
                  f"in-mask={row['cam_in_mask_fraction']:.2f})")


if __name__ == "__main__":
    main()
