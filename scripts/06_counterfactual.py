"""
Step 6: Generate counterfactual explanations.

Usage:
    python3 scripts/06_counterfactual.py
    python3 scripts/06_counterfactual.py --max-per-class 8 --steps 600

What it does:
    - Loads the trained classifier and scores the test set
    - Selects informative examples (confident correct + hard/misclassified)
    - For each image, optimises a perturbation that flips the predicted class
      while keeping changes small and confined to the lung field
    - Saves the counterfactual image, difference map, and comparison panels
    - Produces a summary grid and metadata CSV/JSON
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
import torch.nn.functional as F
from PIL import Image
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
SPLIT_CSV = Path("data/processed/splits/stratified_split.csv")
CHECKPOINT_PATH = Path("experiments/checkpoints/best_classifier.pt")
CF_OUTPUT_DIR = Path("outputs/counterfactuals")
COMP_OUTPUT_DIR = Path("outputs/comparisons")
SELECTED_CSV = Path("outputs/selected_examples.csv")
DECISION_THRESHOLD_PATH = Path("experiments/results/decision_threshold.json")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CURRENT_PREPROCESS_MODE = "baseline"


def confidence_band(confidence):
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "borderline"


def apply_image_preprocessing(image, mode="baseline"):
    image = image.convert("RGB")
    if mode == "baseline":
        return image
    gray = np.asarray(image.convert("L"))
    if mode == "clahe_denoise":
        gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    elif mode != "clahe":
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray)).convert("RGB")


def load_preprocess_mode_from_checkpoint(checkpoint_path, fallback="baseline"):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return fallback
    return checkpoint.get("config", {}).get("preprocess_mode", fallback)


def load_decision_threshold(fallback=0.5):
    if not DECISION_THRESHOLD_PATH.exists():
        return fallback
    try:
        with open(DECISION_THRESHOLD_PATH) as f:
            return float(json.load(f).get("decision_threshold", fallback))
    except Exception:
        return fallback


def predict_label_from_probs(prob, threshold):
    return int(float(prob[1]) >= threshold)


def clamp_normalized(tensor):
    mean = IMAGENET_MEAN.to(tensor.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(tensor.device).view(1, 3, 1, 1)
    lower = (torch.zeros_like(mean) - mean) / std
    upper = (torch.ones_like(mean) - mean) / std
    return torch.max(torch.min(tensor, upper), lower)


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform, split_csv=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples, self.labels = [], []
        if split_csv and Path(split_csv).exists():
            with open(split_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if row["split"] == split:
                        self.samples.append(Path(row["path"]))
                        self.labels.append(CLASS_NAMES.index(row["class"]))
        else:
            for label_idx, class_name in enumerate(CLASS_NAMES):
                class_dir = self.root_dir / split / class_name
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
        transforms.Lambda(lambda img: apply_image_preprocessing(img, CURRENT_PREPROCESS_MODE)),
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
            f"Missing checkpoint: {CHECKPOINT_PATH}. "
            "Run scripts/03_train_classifier.py first."
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


def score_test_set(model, dataset, device, decision_threshold):
    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers)
    rows = []
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Scoring test set"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.array([predict_label_from_probs(prob, decision_threshold) for prob in probs])
            for path, true_idx, pred_idx, prob in zip(
                paths, labels.numpy(), preds, probs
            ):
                rows.append({
                    "path": path,
                    "file": Path(path).name,
                    "true_idx": int(true_idx),
                    "true_label": CLASS_NAMES[int(true_idx)],
                    "pred_idx": int(pred_idx),
                    "pred_label": CLASS_NAMES[int(pred_idx)],
                    "confidence": float(prob[pred_idx]),
                    "confidence_band": confidence_band(float(prob[pred_idx])),
                    "normal_prob": float(prob[0]),
                    "pneumonia_prob": float(prob[1]),
                    "decision_threshold": float(decision_threshold),
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
            key=lambda r: (r["correct"], r["confidence"]),
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
    img = apply_image_preprocessing(Image.open(path).convert("RGB"), CURRENT_PREPROCESS_MODE)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


def write_selected_examples(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_selected_examples(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["true_idx"] = int(row["true_idx"])
            row["pred_idx"] = int(row["pred_idx"])
            for key in ["confidence", "normal_prob", "pneumonia_prob"]:
                row[key] = float(row[key])
            if "true_class_prob" in row:
                row["true_class_prob"] = float(row["true_class_prob"])
            row["correct"] = str(row["correct"]).lower() == "true"
            rows.append(row)
    return rows


def resolve_selected_examples(all_rows, selected_csv, max_per_class, force_reselect=False):
    if not force_reselect:
        selected = load_selected_examples(selected_csv)
        if selected:
            available = {r["path"] for r in all_rows}
            if all(r["path"] in available for r in selected):
                return selected
    selected = select_examples(all_rows, max_per_class)
    write_selected_examples(selected, selected_csv)
    return selected


def make_lung_field_mask(rgb):
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
    candidate = (
        (gray <= dark_cutoff) & (gray >= very_dark) & (body > 0)
    ).astype(np.uint8) * 255

    roi = np.zeros_like(candidate)
    roi[int(IMG_SIZE * 0.12):int(IMG_SIZE * 0.88),
        int(IMG_SIZE * 0.08):int(IMG_SIZE * 0.92)] = 255
    candidate = cv2.bitwise_and(candidate, roi)

    center_band = np.zeros_like(candidate)
    center_band[:, int(IMG_SIZE * 0.42):int(IMG_SIZE * 0.58)] = 255
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(center_band))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
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


def total_variation_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def generate_counterfactual(
    model, input_tensor, source_class, target_class, mask_tensor,
    device, steps, lr, l1_weight, tv_weight, confidence_threshold,
):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros_like(input_tensor, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_delta = None
    best_loss = float("inf")
    best_cf_prob = 0.0
    flipped = False

    for step in range(steps):
        optimizer.zero_grad()
        cf_input = clamp_normalized(input_tensor + delta * mask_tensor)
        logits = model(cf_input)
        probs = F.softmax(logits, dim=1)

        target_prob = probs[0, target_class]
        source_prob = probs[0, source_class]

        cls_loss = -torch.log(target_prob + 1e-8)

        masked_delta = delta * mask_tensor
        l1_loss = masked_delta.abs().mean()
        tv_loss = total_variation_loss(masked_delta)

        loss = cls_loss + l1_weight * l1_loss + tv_weight * tv_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            delta.data.clamp_(-0.5, 0.5)

        current_target_prob = target_prob.item()
        if current_target_prob > best_cf_prob:
            best_cf_prob = current_target_prob
            best_delta = delta.detach().clone()
            best_loss = loss.item()

        if current_target_prob >= confidence_threshold and not flipped:
            flipped = True

    success = best_cf_prob >= 0.5
    return best_delta, success, best_cf_prob, best_loss


def denormalize(tensor):
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return tensor * std + mean


def tensor_to_display(tensor):
    img = denormalize(tensor.squeeze(0)).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def make_difference_map(original_rgb, cf_rgb):
    diff = np.abs(cf_rgb.astype(np.float32) - original_rgb.astype(np.float32))
    diff_gray = diff.mean(axis=2)
    if diff_gray.max() > 0:
        diff_gray = diff_gray / diff_gray.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * diff_gray), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = np.clip(diff_gray, 0, 1) ** 0.6
    blended = original_rgb * (1 - alpha[..., None]) + heatmap * alpha[..., None]
    return diff_gray, np.clip(blended, 0, 1)


def change_localisation_metrics(diff_gray, lung_mask, threshold=0.02):
    lung_mask = (lung_mask > 0.5).astype(np.float32)
    total_change = float(diff_gray.sum())
    if total_change <= 1e-8:
        change_inside_lung = 0.0
    else:
        change_inside_lung = float((diff_gray * lung_mask).sum() / total_change)

    changed = diff_gray > threshold
    if not changed.any():
        active_inside_lung = 0.0
    else:
        active_inside_lung = float(lung_mask[changed].mean())

    return {
        "change_inside_lung_fraction": round(change_inside_lung, 4),
        "change_outside_lung_fraction": round(1.0 - change_inside_lung, 4),
        "active_change_inside_lung_fraction": round(active_inside_lung, 4),
        "active_change_outside_lung_fraction": round(1.0 - active_inside_lung, 4),
    }


def save_counterfactual_panel(
    row, original_rgb, cf_rgb, diff_gray, diff_overlay, mask,
    cf_pred_label, cf_confidence, output_dir, comp_dir, idx,
):
    true_label = row["true_label"].lower()
    stem = f"{idx:02d}_{true_label}_{Path(row['path']).stem}"

    Image.fromarray(np.uint8(cf_rgb * 255)).save(
        output_dir / f"{stem}_counterfactual.png"
    )
    plt.imsave(
        output_dir / f"{stem}_diff_map.png", diff_gray, cmap="turbo", vmin=0, vmax=1
    )

    fig, axes = plt.subplots(1, 5, figsize=(24, 6.5))

    axes[0].imshow(original_rgb)
    axes[0].set_title(
        "1. Original X-ray\n"
        f"True label: {row['true_label']}\n"
        f"Model predicted: {row['pred_label']} ({row['confidence']:.3f})",
        fontsize=10,
    )

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        "2. Where changes were allowed\n"
        "White = lung area the algorithm may edit\n"
        "Black = protected / not changed",
        fontsize=10,
    )

    axes[2].imshow(cf_rgb)
    axes[2].set_title(
        "3. Changed X-ray\n"
        "Synthetic version after tiny edits\n"
        f"New prediction: {cf_pred_label} ({cf_confidence:.3f})",
        fontsize=10,
    )

    axes[3].imshow(diff_gray, cmap="turbo", vmin=0, vmax=1)
    axes[3].set_title(
        "4. Difference map\n"
        "Colors show how much each pixel changed\n"
        "Dark = little/no change, bright = bigger change",
        fontsize=10,
    )

    axes[4].imshow(diff_overlay)
    axes[4].set_title(
        "5. Changes drawn on X-ray\n"
        "Same changes drawn on the original X-ray\n"
        "Colored areas drove the prediction flip",
        fontsize=10,
    )

    for ax in axes:
        ax.axis("off")

    status = "correct" if row["correct"] else "misclassified"
    fig.suptitle(
        f"What small change would flip the model for {Path(row['path']).name}  |  "
        f"{status} original prediction  |  model changed: {row['pred_label']} -> {cf_pred_label}",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "How to read this: the algorithm asks, 'What is the smallest visible change inside the lung/chest area "
        "that would make the model switch its answer?' This is not a real patient image; it is a "
        "synthetic explanation of what the trained model is sensitive to.",
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    panel_path = comp_dir / f"{stem}_cf_panel.png"
    plt.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def save_summary_grid(results, comp_dir):
    successful = [r for r in results if r["cf_success"]]
    if not successful:
        print("  No successful counterfactuals to plot in summary grid.")
        return

    cols = 3
    n = len(successful)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 8, rows * 4.2))
    axes = np.atleast_2d(axes).reshape(rows, cols * 2)

    for ax in axes.reshape(-1):
        ax.axis("off")

    for i, result in enumerate(successful):
        row_idx = i // cols
        col_idx = i % cols

        ax_orig = axes[row_idx, col_idx * 2]
        ax_cf = axes[row_idx, col_idx * 2 + 1]

        ax_orig.imshow(result["original_rgb"])
        ax_orig.set_title(
            f"Original: {result['pred_label']}\n({result['confidence']:.2f})",
            fontsize=8,
        )

        ax_cf.imshow(result["cf_rgb"])
        ax_cf.set_title(
            f"Changed image: {result['cf_pred_label']}\n({result['cf_confidence']:.2f})",
            fontsize=8,
        )

        ax_orig.axis("off")
        ax_cf.axis("off")

    plt.suptitle(
        "Original X-rays vs Synthetic Changed X-rays",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(comp_dir / "counterfactual_summary_grid.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def save_difference_grid(results, comp_dir):
    successful = [r for r in results if r["cf_success"]]
    if not successful:
        return

    n = len(successful)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 3.8))
    axes = np.atleast_2d(axes)

    for i, result in enumerate(successful):
        axes[i, 0].imshow(result["original_rgb"])
        axes[i, 0].set_title(
            f"Original: {result['pred_label']} ({result['confidence']:.2f})",
            fontsize=8,
        )
        axes[i, 1].imshow(result["diff_overlay"])
        axes[i, 1].set_title(
            "Colored areas = pixels changed\n"
            "to flip the model answer",
            fontsize=8,
        )
        axes[i, 2].imshow(result["cf_rgb"])
        axes[i, 2].set_title(
            f"Changed image: {result['cf_pred_label']}\n"
            f"({result['cf_confidence']:.2f})",
            fontsize=8,
        )
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(
        "What would need to change to flip the model's diagnosis?",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comp_dir / "counterfactual_diff_grid.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def write_metadata(results, all_rows, cf_dir, comp_dir, args):
    serialisable = []
    for r in results:
        entry = {k: v for k, v in r.items()
                 if k not in {"original_rgb", "cf_rgb", "diff_gray", "diff_overlay"}}
        serialisable.append(entry)

    metadata = {
        "optimisation": {
            "steps": args.steps,
            "lr": args.lr,
            "l1_weight": args.l1_weight,
            "tv_weight": args.tv_weight,
            "confidence_threshold": args.confidence_threshold,
        },
        "max_per_class": args.max_per_class,
        "preprocess_mode": results[0].get("preprocess_mode", "") if results else "",
        "selected_examples_csv": str(SELECTED_CSV),
        "selected_count": len(results),
        "successful_count": sum(1 for r in results if r["cf_success"]),
        "all_test_count": len(all_rows),
        "selected": serialisable,
    }
    with open(cf_dir / "counterfactual_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    fieldnames = [
        "file", "path", "true_label", "pred_label", "correct", "confidence",
        "confidence_band", "cf_pred_label", "cf_confidence", "cf_success",
        "cf_target_prob", "cf_l1_norm", "change_inside_lung_fraction",
        "active_change_outside_lung_fraction", "cf_steps_used", "panel_path",
    ]
    with open(cf_dir / "counterfactual_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({name: r.get(name) for name in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual explanations for the pneumonia classifier."
    )
    parser.add_argument("--max-per-class", type=int, default=6,
                        help="Test examples per true class.")
    parser.add_argument("--steps", type=int, default=500,
                        help="Optimisation steps per image.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for the perturbation.")
    parser.add_argument("--l1-weight", type=float, default=0.008,
                        help="L1 sparsity penalty on the perturbation.")
    parser.add_argument("--tv-weight", type=float, default=0.003,
                        help="Total-variation smoothness penalty.")
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Target-class probability to consider the CF successful.")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep old output files.")
    parser.add_argument("--reselect", action="store_true",
                        help="Force a new shared selected_examples.csv for all XAI steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    CF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_clean:
        for d in [CF_OUTPUT_DIR, COMP_OUTPUT_DIR]:
            for path in d.glob("*"):
                if path.is_file():
                    path.unlink()

    device = get_device()
    print(f"Using device: {device}")

    global CURRENT_PREPROCESS_MODE
    preprocess_mode = load_preprocess_mode_from_checkpoint(CHECKPOINT_PATH, fallback="baseline")
    CURRENT_PREPROCESS_MODE = preprocess_mode
    transform = build_eval_transform()
    split_csv = SPLIT_CSV if SPLIT_CSV.exists() else None
    dataset = ChestXRayDataset(DATA_DIR, "test", transform, split_csv=split_csv)
    if len(dataset) == 0:
        raise SystemExit(
            f"No test images found under {DATA_DIR}. "
            "Run scripts/01_download_data.py first."
        )

    model = build_model(device)
    decision_threshold = load_decision_threshold()
    all_rows = score_test_set(model, dataset, device, decision_threshold)
    selected = resolve_selected_examples(
        all_rows, SELECTED_CSV, args.max_per_class, force_reselect=args.reselect
    )
    print(
        f"Selected {len(selected)} examples from {len(all_rows)} test images"
    )
    print(f"Preprocessing: {preprocess_mode}")
    print(f"Decision threshold: {decision_threshold:.3f}")

    mean_t = IMAGENET_MEAN.to(device)
    std_t = IMAGENET_STD.to(device)

    results = []
    for idx, row in enumerate(tqdm(selected, desc="Generating counterfactuals")):
        rgb = load_display_image(row["path"])
        mask_np = make_lung_field_mask(rgb)
        mask_tensor = (
            torch.from_numpy(mask_np)
            .unsqueeze(0).unsqueeze(0)
            .expand(1, 3, -1, -1)
            .to(device)
        )

        input_tensor = (
            transform(Image.open(row["path"]).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )

        source_class = row["pred_idx"]
        target_class = 1 - source_class

        best_delta, success, cf_target_prob, cf_loss = generate_counterfactual(
            model=model,
            input_tensor=input_tensor,
            source_class=source_class,
            target_class=target_class,
            mask_tensor=mask_tensor,
            device=device,
            steps=args.steps,
            lr=args.lr,
            l1_weight=args.l1_weight,
            tv_weight=args.tv_weight,
            confidence_threshold=args.confidence_threshold,
        )

        with torch.no_grad():
            cf_tensor = clamp_normalized(input_tensor + best_delta * mask_tensor)
            cf_logits = model(cf_tensor)
            cf_probs = F.softmax(cf_logits, dim=1).cpu().numpy()[0]
            cf_pred_idx = predict_label_from_probs(cf_probs, decision_threshold)
            cf_confidence = float(cf_probs[cf_pred_idx])
            cf_pred_label = CLASS_NAMES[cf_pred_idx]

        cf_rgb = tensor_to_display(cf_tensor)
        original_rgb = tensor_to_display(input_tensor)
        diff_gray, diff_overlay = make_difference_map(original_rgb, cf_rgb)
        localisation = change_localisation_metrics(diff_gray, mask_np)

        masked_delta_np = (best_delta * mask_tensor).detach().cpu().numpy()
        l1_norm = float(np.abs(masked_delta_np).mean())

        panel_path = save_counterfactual_panel(
            row, original_rgb, cf_rgb, diff_gray, diff_overlay, mask_np,
            cf_pred_label, cf_confidence, CF_OUTPUT_DIR, COMP_OUTPUT_DIR, idx,
        )

        result = {
            **row,
            "cf_pred_idx": cf_pred_idx,
            "cf_pred_label": cf_pred_label,
            "cf_confidence": cf_confidence,
            "cf_success": success,
            "cf_target_prob": cf_target_prob,
            "cf_l1_norm": l1_norm,
            **localisation,
            "cf_steps_used": args.steps,
            "preprocess_mode": preprocess_mode,
            "panel_path": str(panel_path),
            "original_rgb": original_rgb,
            "cf_rgb": cf_rgb,
            "diff_gray": diff_gray,
            "diff_overlay": diff_overlay,
        }
        results.append(result)

        status = "FLIPPED" if success else "PARTIAL"
        print(
            f"  [{status}] {row['file']}: "
            f"{row['pred_label']}({row['confidence']:.2f}) -> "
            f"{cf_pred_label}({cf_confidence:.2f}), "
            f"L1={l1_norm:.4f}"
        )

    save_summary_grid(results, COMP_OUTPUT_DIR)
    save_difference_grid(results, COMP_OUTPUT_DIR)
    write_metadata(results, all_rows, CF_OUTPUT_DIR, COMP_OUTPUT_DIR, args)

    n_success = sum(1 for r in results if r["cf_success"])
    n_total = len(results)

    print(f"\nCounterfactual generation complete.")
    print(f"  Success rate: {n_success}/{n_total} "
          f"({100 * n_success / max(n_total, 1):.0f}%)")
    print("\nStep 6 outputs")
    print(f"  Counterfactual directory: {CF_OUTPUT_DIR}")
    print(f"  Comparison directory:     {COMP_OUTPUT_DIR}")
    print(f"  Metadata JSON:            {CF_OUTPUT_DIR / 'counterfactual_metadata.json'}")
    print(f"  Metadata CSV:             {CF_OUTPUT_DIR / 'counterfactual_metadata.csv'}")
    print(f"  Summary grid:             {COMP_OUTPUT_DIR / 'counterfactual_summary_grid.png'}")
    print(f"  Difference grid:          {COMP_OUTPUT_DIR / 'counterfactual_diff_grid.png'}")
    print(f"  Per-image CF images:      {CF_OUTPUT_DIR / '*_counterfactual.png'}")
    print(f"  Per-image difference maps:{CF_OUTPUT_DIR / '*_diff_map.png'}")
    print(f"  Per-image panels:         {COMP_OUTPUT_DIR / '*_cf_panel.png'}")

    if n_success < n_total:
        partial = [r for r in results if not r["cf_success"]]
        print(f"\n  Partial flips ({len(partial)}) — try increasing --steps or --lr:")
        for r in partial[:5]:
            print(
                f"    - {r['file']}: target_prob reached {r['cf_target_prob']:.3f}"
            )


if __name__ == "__main__":
    main()
