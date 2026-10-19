"""
Step 8: Evaluate explanation quality with quantitative metrics.

Usage:
    python scripts/08_explanation_quality.py
    python scripts/08_explanation_quality.py --max-per-class 8

What it does:
    - Measures how faithful and precise the explanations from steps 05-07 are
    - For Grad-CAM: insertion/deletion metrics (does masking the highlighted
      region actually change the prediction?), in-mask fraction, sparsity
    - For Counterfactuals: SSIM (structural similarity to original), L1 distance,
      perturbation sparsity, prediction flip success rate
    - For Guided Counterfactuals: same metrics plus comparison to standard CFs
    - For GAN Counterfactuals: the same image-change and lung-localisation audit,
      reported overall and separately for both generation directions
    - These metrics answer: "Is the model actually using the regions the explanation
      highlights?" and "Are the counterfactual changes minimal and realistic?"
    - Saves metrics to JSON and generates comparison bar charts
"""

import json
import os
import argparse
import csv
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.metrics import structural_similarity as ssim
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
RESULTS_DIR = Path("experiments/results")
GRADCAM_DIR = Path("outputs/gradcam")
CF_DIR = Path("outputs/counterfactuals/classic")
GAN_CF_DIR = Path("outputs/counterfactuals/gan")
GUIDED_CF_DIR = Path("outputs/guided_counterfactuals")
OUTPUT_DIR = Path("outputs/explanation_quality")
SELECTED_CSV = Path("outputs/selected_examples.csv")
DECISION_THRESHOLD_PATH = RESULTS_DIR / "decision_threshold.json"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CURRENT_PREPROCESS_MODE = "baseline"


def clean_generated_outputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def trapezoid_auc(values):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values)
    return np.trapz(values)


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


def load_display_image(path):
    img = apply_image_preprocessing(Image.open(path).convert("RGB"), CURRENT_PREPROCESS_MODE)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


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


def anatomical_localisation_metrics(signal, lung_mask, threshold=0.3):
    signal = np.asarray(signal, dtype=np.float32)
    lung_mask = (np.asarray(lung_mask) > 0.5).astype(np.float32)
    total_signal = float(signal.sum())
    if total_signal <= 1e-8:
        signal_inside_lung = 0.0
        signal_outside_lung = 0.0
    else:
        signal_inside_lung = float((signal * lung_mask).sum() / total_signal)
        signal_outside_lung = 1.0 - signal_inside_lung

    active = signal >= threshold
    active_count = int(active.sum())
    if active_count == 0:
        active_inside_lung = 0.0
        active_outside_lung = 0.0
    else:
        active_inside_lung = float(lung_mask[active].mean())
        active_outside_lung = 1.0 - active_inside_lung

    return {
        "signal_inside_lung_fraction": round(signal_inside_lung, 4),
        "signal_outside_lung_fraction": round(signal_outside_lung, 4),
        "active_inside_lung_fraction": round(active_inside_lung, 4),
        "active_outside_lung_fraction": round(active_outside_lung, 4),
    }


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


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform, split_csv=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples, self.labels = [], []
        if split_csv and Path(split_csv).exists():
            import csv
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


def write_selected_examples(rows, path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_selected_examples(path):
    import csv
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["true_idx"] = int(row["true_idx"])
            row["pred_idx"] = int(row["pred_idx"])
            for key in ["confidence"]:
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
                    "decision_threshold": float(decision_threshold),
                    "correct": bool(pred_idx == true_idx),
                })
    return rows


def select_examples(rows, max_per_class):
    selected = []
    seen = set()
    for class_idx in range(NUM_CLASSES):
        class_rows = [r for r in rows if r["true_idx"] == class_idx]
        correct = sorted(
            [r for r in class_rows if r["correct"]],
            key=lambda r: r["confidence"], reverse=True,
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


def insertion_deletion_test(model, input_tensor, cam, target_class, device, steps=20):
    """
    Insertion: progressively reveal pixels from most to least important.
    Deletion: progressively remove pixels from most to least important.
    A faithful explanation should show: insertion raises confidence, deletion drops it.
    """
    model.eval()
    cam_flat = cam.flatten()
    sorted_indices = np.argsort(-cam_flat)
    n_pixels = len(cam_flat)
    step_size = max(1, n_pixels // steps)

    mean_tensor = IMAGENET_MEAN.to(device).unsqueeze(0).expand_as(input_tensor)
    blurred = mean_tensor.clone()

    insertion_scores = []
    deletion_scores = []

    insertion_mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    deletion_mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)

    for step in range(steps + 1):
        end_idx = min(step * step_size, n_pixels)

        if step > 0:
            pixel_indices = sorted_indices[(step - 1) * step_size:end_idx]
            rows_idx = pixel_indices // IMG_SIZE
            cols_idx = pixel_indices % IMG_SIZE
            insertion_mask[0, 0, rows_idx, cols_idx] = 1.0
            deletion_mask[0, 0, rows_idx, cols_idx] = 0.0

        ins_input = input_tensor * insertion_mask + blurred * (1 - insertion_mask)
        del_input = input_tensor * deletion_mask + blurred * (1 - deletion_mask)

        with torch.no_grad():
            ins_prob = torch.softmax(model(ins_input), dim=1)[0, target_class].item()
            del_prob = torch.softmax(model(del_input), dim=1)[0, target_class].item()

        insertion_scores.append(ins_prob)
        deletion_scores.append(del_prob)

    insertion_auc = float(trapezoid_auc(insertion_scores) / len(insertion_scores))
    deletion_auc = float(trapezoid_auc(deletion_scores) / len(deletion_scores))

    return {
        "insertion_auc": round(insertion_auc, 4),
        "deletion_auc": round(deletion_auc, 4),
        "faithfulness": round(insertion_auc - deletion_auc, 4),
        "insertion_curve": [round(s, 4) for s in insertion_scores],
        "deletion_curve": [round(s, 4) for s in deletion_scores],
    }


def compute_counterfactual_metrics(original_rgb, cf_rgb, lung_mask=None):
    ssim_val = ssim(
        original_rgb, cf_rgb,
        channel_axis=2,
        data_range=1.0,
    )
    l1_dist = float(np.abs(original_rgb - cf_rgb).mean())
    l2_dist = float(np.sqrt(np.mean((original_rgb - cf_rgb) ** 2)))

    diff = np.abs(original_rgb - cf_rgb).mean(axis=2)
    changed_pixels = float((diff > 0.02).mean())

    metrics = {
        "ssim": round(float(ssim_val), 4),
        "l1_distance": round(l1_dist, 6),
        "l2_distance": round(l2_dist, 6),
        "changed_pixel_fraction": round(changed_pixels, 4),
        "sparsity": round(1.0 - changed_pixels, 4),
    }
    if lung_mask is not None:
        metrics.update(anatomical_localisation_metrics(diff, lung_mask, threshold=0.02))
    return metrics


def evaluate_gradcam(model, selected, transform, device):
    print("\nEvaluating Grad-CAM quality...")
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    results = []

    for row in tqdm(selected, desc="Grad-CAM fidelity"):
        original_rgb = load_display_image(row["path"])
        lung_mask = make_lung_field_mask(original_rgb)
        input_tensor = transform(
            Image.open(row["path"]).convert("RGB")
        ).unsqueeze(0).to(device)

        target_idx = CLASS_NAMES.index("PNEUMONIA")
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_idx)],
        )[0]
        grayscale_cam = percentile_normalize(grayscale_cam, lung_mask)

        fidelity = insertion_deletion_test(
            model, input_tensor, grayscale_cam, target_idx, device,
        )

        cam_sparsity = float((grayscale_cam < 0.1).mean())
        cam_peak = float(grayscale_cam.max())
        cam_coverage = float((grayscale_cam >= 0.3).mean())
        anatomy = anatomical_localisation_metrics(
            grayscale_cam, lung_mask, threshold=0.3,
        )

        results.append({
            "file": row["file"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "confidence_band": row.get("confidence_band", confidence_band(row["confidence"])),
            **fidelity,
            "cam_sparsity": round(cam_sparsity, 4),
            "cam_peak": round(cam_peak, 4),
            "cam_coverage_30pct": round(cam_coverage, 4),
            "cam_signal_inside_lung_fraction": anatomy["signal_inside_lung_fraction"],
            "cam_signal_outside_lung_fraction": anatomy["signal_outside_lung_fraction"],
            "cam_active_inside_lung_fraction": anatomy["active_inside_lung_fraction"],
            "cam_active_outside_lung_fraction": anatomy["active_outside_lung_fraction"],
        })

    return results


def evaluate_counterfactuals(cf_dir, label):
    meta_path = cf_dir / "counterfactual_metadata.json"
    if not meta_path.exists():
        print(f"  {label}: metadata not found at {meta_path}, skipping.")
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    results = []
    for idx, entry in enumerate(meta.get("selected", [])):
        path = entry.get("path", "")
        file_stem = Path(path).stem
        true_label = entry.get("true_label", "unknown").lower()
        desc_stem = f"{idx + 1:02d}_{true_label}_xray"
        cf_candidates = (
            list(cf_dir.glob(f"*{file_stem}*_edited_to_look_*.png"))
            or list(cf_dir.glob(f"*{file_stem}*_counterfactual.png"))
            or list(cf_dir.glob(f"*{desc_stem}*_edited_to_look_*.png"))
        )
        if not cf_candidates or not Path(path).exists():
            continue

        original = load_display_image(path)
        cf_img = np.asarray(
            Image.open(cf_candidates[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0
        lung_mask = make_lung_field_mask(original)

        metrics = compute_counterfactual_metrics(original, cf_img, lung_mask)
        results.append({
            "file": entry.get("file", ""),
            "true_label": entry.get("true_label", ""),
            "pred_label": entry.get("pred_label", ""),
            "confidence": entry.get("confidence", 0),
            "confidence_band": confidence_band(entry.get("confidence", 0)),
            "cf_pred_label": entry.get("cf_pred_label", ""),
            "cf_success": entry.get("cf_success", False),
            "cf_confidence": entry.get("cf_confidence", 0),
            **metrics,
        })

    return results


def evaluate_guided_counterfactuals(guided_dir):
    meta_path = guided_dir / "guided_cf_metadata.json"
    if not meta_path.exists():
        print(f"  Guided CF: metadata not found at {meta_path}, skipping.")
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    results = []
    for idx, entry in enumerate(meta.get("selected", [])):
        path = entry.get("path", "")
        file_stem = Path(path).stem
        true_label = entry.get("true_label", "unknown").lower()
        desc_stem = f"{idx + 1:02d}_{true_label}_xray"
        cf_candidates = (
            list(guided_dir.glob(f"*{file_stem}*_guided_what_if_*.png"))
            or list(guided_dir.glob(f"*{file_stem}*_guided_cf.png"))
            or list(guided_dir.glob(f"*{desc_stem}*_guided_cf.png"))
            or list(guided_dir.glob(f"*{desc_stem}*_guided_what_if_*.png"))
        )
        if not cf_candidates or not Path(path).exists():
            continue

        original = load_display_image(path)
        cf_img = np.asarray(
            Image.open(cf_candidates[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0
        lung_mask = make_lung_field_mask(original)

        metrics = compute_counterfactual_metrics(original, cf_img, lung_mask)
        results.append({
            "file": entry.get("file", ""),
            "true_label": entry.get("true_label", ""),
            "pred_label": entry.get("pred_label", ""),
            "confidence": entry.get("confidence", 0),
            "confidence_band": confidence_band(entry.get("confidence", 0)),
            "guided_cf_pred_label": entry.get("guided_cf_pred_label", ""),
            "guided_cf_success": entry.get("guided_cf_success", False),
            "guided_cf_confidence": entry.get("guided_cf_confidence", 0),
            "standard_l1": entry.get("standard_l1", 0),
            "guided_l1": entry.get("guided_l1", 0),
            "l1_reduction_pct": entry.get("l1_reduction_pct", 0),
            **metrics,
        })

    return results


def evaluate_gan_counterfactuals(gan_dir):
    metrics_path = gan_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"  GAN CF: metrics not found at {metrics_path}, skipping.")
        return []

    results = []
    with open(metrics_path, newline="") as handle:
        entries = list(csv.DictReader(handle))

    for entry in entries:
        source_path = Path(entry.get("source_path", ""))
        case = int(entry.get("case", 0))
        direction = entry.get("direction", "")
        generated_path = gan_dir / f"case_{case:02d}_{direction.lower()}.png"
        if not source_path.exists() or not generated_path.exists():
            continue

        original = load_display_image(source_path)
        generated = np.asarray(
            Image.open(generated_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0
        lung_mask = make_lung_field_mask(original)
        metrics = compute_counterfactual_metrics(original, generated, lung_mask)
        results.append({
            "case": case,
            "file": source_path.name,
            "source_path": str(source_path),
            "direction": direction,
            "success": entry.get("success", "").lower() == "true",
            "target_probability_after": float(entry.get("target_probability_after", 0)),
            "applied_strength": float(entry.get("applied_strength", 0)),
            **metrics,
        })

    return results


def save_fidelity_chart(gradcam_results, output_dir):
    if not gradcam_results:
        return

    files = [r["file"][:20] for r in gradcam_results]
    ins_aucs = [r["insertion_auc"] for r in gradcam_results]
    del_aucs = [r["deletion_auc"] for r in gradcam_results]
    faithfulness = [r["faithfulness"] for r in gradcam_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(files))
    w = 0.35
    axes[0].bar(x - w / 2, ins_aucs, w, label="Reveal important pixels", color="#22c55e")
    axes[0].bar(x + w / 2, del_aucs, w, label="Remove important pixels", color="#ef4444")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Area under curve")
    axes[0].set_title(
        "Does the focus map point to useful pixels?\n"
        "Good = prediction rises when focus pixels are revealed and drops when removed"
    )
    axes[0].legend()

    colors = ["#22c55e" if f > 0.3 else "#f59e0b" if f > 0.1 else "#ef4444"
              for f in faithfulness]
    axes[1].bar(x, faithfulness, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Usefulness score")
    axes[1].set_title(
        "Overall focus-map usefulness\n"
        "Green = stronger evidence, red = weak or unreliable"
    )
    axes[1].axhline(y=0.3, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_fidelity_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_counterfactual_quality_chart(cf_results, guided_results, gan_results, output_dir):
    if not cf_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cf_ssim = [r["ssim"] for r in cf_results]
    cf_sparsity = [r["sparsity"] for r in cf_results]
    cf_l1 = [r["l1_distance"] for r in cf_results]

    labels = ["Classic CF"]
    ssim_means = [np.mean(cf_ssim)]
    sparsity_means = [np.mean(cf_sparsity)]
    l1_means = [np.mean(cf_l1)]

    if guided_results:
        g_ssim = [r["ssim"] for r in guided_results]
        g_sparsity = [r["sparsity"] for r in guided_results]
        g_l1 = [r["l1_distance"] for r in guided_results]
        labels.append("Guided CF")
        ssim_means.append(np.mean(g_ssim))
        sparsity_means.append(np.mean(g_sparsity))
        l1_means.append(np.mean(g_l1))

    if gan_results:
        gan_ssim = [r["ssim"] for r in gan_results]
        gan_sparsity = [r["sparsity"] for r in gan_results]
        gan_l1 = [r["l1_distance"] for r in gan_results]
        labels.append("GAN CF")
        ssim_means.append(np.mean(gan_ssim))
        sparsity_means.append(np.mean(gan_sparsity))
        l1_means.append(np.mean(gan_l1))

    colors = ["#3b82f6", "#22c55e", "#f59e0b"][:len(labels)]

    axes[0].bar(labels, ssim_means, color=colors)
    axes[0].set_title(
        "Similarity to original X-ray\n"
        "Higher = changed image still looks close to the original"
    )
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(ssim_means):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[1].bar(labels, sparsity_means, color=colors)
    axes[1].set_title(
        "How many pixels stayed unchanged\n"
        "Higher = fewer pixels changed, easier to explain"
    )
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(sparsity_means):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[2].bar(labels, l1_means, color=colors)
    axes[2].set_title(
        "Total change size\n"
        "Lower = smaller synthetic edit was enough to flip the model"
    )
    for i, v in enumerate(l1_means):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "counterfactual_quality_chart.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_anatomical_plausibility_chart(
    gradcam_results, cf_results, guided_results, gan_results, output_dir
):
    labels = []
    inside_signal = []
    outside_active = []

    if gradcam_results:
        labels.append("Grad-CAM++")
        inside_signal.append(np.mean([
            r["cam_signal_inside_lung_fraction"] for r in gradcam_results
        ]))
        outside_active.append(np.mean([
            r["cam_active_outside_lung_fraction"] for r in gradcam_results
        ]))

    if cf_results:
        labels.append("Classic CF")
        inside_signal.append(np.mean([
            r["signal_inside_lung_fraction"] for r in cf_results
        ]))
        outside_active.append(np.mean([
            r["active_outside_lung_fraction"] for r in cf_results
        ]))

    if guided_results:
        labels.append("Guided CF")
        inside_signal.append(np.mean([
            r["signal_inside_lung_fraction"] for r in guided_results
        ]))
        outside_active.append(np.mean([
            r["active_outside_lung_fraction"] for r in guided_results
        ]))

    if gan_results:
        labels.append("GAN CF")
        inside_signal.append(np.mean([
            r["signal_inside_lung_fraction"] for r in gan_results
        ]))
        outside_active.append(np.mean([
            r["active_outside_lung_fraction"] for r in gan_results
        ]))

    if not labels:
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(x, inside_signal, color="#2563eb")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of explanation")
    axes[0].set_title(
        "How much explanation stays inside the chest/lung area?\n"
        "Higher is better because pneumonia evidence should be in the chest"
    )
    for i, v in enumerate(inside_signal):
        axes[0].text(i, min(v + 0.03, 0.98), f"{v:.2f}", ha="center")

    axes[1].bar(x, outside_active, color="#dc2626")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of explanation")
    axes[1].set_title(
        "How much explanation leaks outside the chest/lung area?\n"
        "Lower is better because outside regions are less medically useful"
    )
    for i, v in enumerate(outside_active):
        axes[1].text(i, min(v + 0.03, 0.98), f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "anatomical_plausibility_chart.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_insertion_deletion_curves(gradcam_results, output_dir):
    if not gradcam_results:
        return

    n = min(len(gradcam_results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.reshape(-1)

    for i in range(6):
        if i >= n:
            axes[i].axis("off")
            continue

        r = gradcam_results[i]
        steps_x = np.linspace(0, 1, len(r["insertion_curve"]))
        axes[i].plot(steps_x, r["insertion_curve"], "g-", linewidth=2, label="Reveal focus pixels")
        axes[i].plot(steps_x, r["deletion_curve"], "r-", linewidth=2, label="Remove focus pixels")
        axes[i].fill_between(steps_x, r["insertion_curve"], r["deletion_curve"],
                             alpha=0.15, color="blue")
        axes[i].set_xlabel("Amount of focus area tested")
        axes[i].set_ylabel("Model confidence")
        status = "OK" if r["correct"] else "MISS"
        axes[i].set_title(
            f"{status} {r['file'][:25]}\nUsefulness={r['faithfulness']:.3f}",
            fontsize=9,
        )
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(-0.05, 1.05)

    plt.suptitle(
        "Focus-map test: does removing highlighted areas reduce model confidence?",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "insertion_deletion_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM and counterfactual explanation quality."
    )
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep old output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_clean:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        clean_generated_outputs(OUTPUT_DIR)

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
        all_rows, SELECTED_CSV, max_per_class=6, force_reselect=False
    )
    print(f"Preprocessing: {preprocess_mode}")
    print(f"Decision threshold: {decision_threshold:.3f}")

    gradcam_results = evaluate_gradcam(model, selected, transform, device)

    print("\nEvaluating standard counterfactual quality...")
    cf_results = evaluate_counterfactuals(CF_DIR, "Standard CF")

    print("Evaluating guided counterfactual quality...")
    guided_results = evaluate_guided_counterfactuals(GUIDED_CF_DIR)

    print("Evaluating GAN counterfactual quality...")
    gan_results = evaluate_gan_counterfactuals(GAN_CF_DIR)

    save_fidelity_chart(gradcam_results, OUTPUT_DIR)
    save_insertion_deletion_curves(gradcam_results, OUTPUT_DIR)
    save_counterfactual_quality_chart(
        cf_results, guided_results, gan_results, OUTPUT_DIR,
    )
    save_anatomical_plausibility_chart(
        gradcam_results, cf_results, guided_results, gan_results, OUTPUT_DIR,
    )

    summary = {
        "gradcam": {
            "num_evaluated": len(gradcam_results),
            "mean_insertion_auc": round(float(np.mean(
                [r["insertion_auc"] for r in gradcam_results]
            )), 4) if gradcam_results else 0,
            "mean_deletion_auc": round(float(np.mean(
                [r["deletion_auc"] for r in gradcam_results]
            )), 4) if gradcam_results else 0,
            "mean_faithfulness": round(float(np.mean(
                [r["faithfulness"] for r in gradcam_results]
            )), 4) if gradcam_results else 0,
            "mean_cam_signal_inside_lung_fraction": round(float(np.mean(
                [r["cam_signal_inside_lung_fraction"] for r in gradcam_results]
            )), 4) if gradcam_results else 0,
            "mean_cam_active_outside_lung_fraction": round(float(np.mean(
                [r["cam_active_outside_lung_fraction"] for r in gradcam_results]
            )), 4) if gradcam_results else 0,
            "per_image": gradcam_results,
        },
        "standard_counterfactual": {
            "num_evaluated": len(cf_results),
            "mean_ssim": round(float(np.mean(
                [r["ssim"] for r in cf_results]
            )), 4) if cf_results else 0,
            "mean_l1": round(float(np.mean(
                [r["l1_distance"] for r in cf_results]
            )), 6) if cf_results else 0,
            "mean_sparsity": round(float(np.mean(
                [r["sparsity"] for r in cf_results]
            )), 4) if cf_results else 0,
            "mean_change_inside_lung_fraction": round(float(np.mean(
                [r["signal_inside_lung_fraction"] for r in cf_results]
            )), 4) if cf_results else 0,
            "mean_active_change_outside_lung_fraction": round(float(np.mean(
                [r["active_outside_lung_fraction"] for r in cf_results]
            )), 4) if cf_results else 0,
            "success_rate": round(
                sum(1 for r in cf_results if r["cf_success"]) / max(len(cf_results), 1), 4
            ) if cf_results else 0,
            "per_image": cf_results,
        },
        "guided_counterfactual": {
            "num_evaluated": len(guided_results),
            "mean_ssim": round(float(np.mean(
                [r["ssim"] for r in guided_results]
            )), 4) if guided_results else 0,
            "mean_l1": round(float(np.mean(
                [r["l1_distance"] for r in guided_results]
            )), 6) if guided_results else 0,
            "mean_sparsity": round(float(np.mean(
                [r["sparsity"] for r in guided_results]
            )), 4) if guided_results else 0,
            "mean_change_inside_lung_fraction": round(float(np.mean(
                [r["signal_inside_lung_fraction"] for r in guided_results]
            )), 4) if guided_results else 0,
            "mean_active_change_outside_lung_fraction": round(float(np.mean(
                [r["active_outside_lung_fraction"] for r in guided_results]
            )), 4) if guided_results else 0,
            "success_rate": round(
                sum(1 for r in guided_results if r.get("guided_cf_success", False)) /
                max(len(guided_results), 1), 4
            ) if guided_results else 0,
            "per_image": guided_results,
        },
        "gan_counterfactual": {
            "num_evaluated": len(gan_results),
            "mean_ssim": round(float(np.mean(
                [r["ssim"] for r in gan_results]
            )), 4) if gan_results else 0,
            "mean_l1": round(float(np.mean(
                [r["l1_distance"] for r in gan_results]
            )), 6) if gan_results else 0,
            "mean_sparsity": round(float(np.mean(
                [r["sparsity"] for r in gan_results]
            )), 4) if gan_results else 0,
            "mean_change_inside_lung_fraction": round(float(np.mean(
                [r["signal_inside_lung_fraction"] for r in gan_results]
            )), 4) if gan_results else 0,
            "mean_active_change_outside_lung_fraction": round(float(np.mean(
                [r["active_outside_lung_fraction"] for r in gan_results]
            )), 4) if gan_results else 0,
            "success_rate": round(
                sum(1 for r in gan_results if r.get("success", False)) /
                max(len(gan_results), 1), 4
            ) if gan_results else 0,
            "by_direction": {
                direction: {
                    "num_evaluated": len(direction_rows),
                    "mean_ssim": round(float(np.mean(
                        [r["ssim"] for r in direction_rows]
                    )), 4),
                    "mean_l1": round(float(np.mean(
                        [r["l1_distance"] for r in direction_rows]
                    )), 6),
                    "mean_sparsity": round(float(np.mean(
                        [r["sparsity"] for r in direction_rows]
                    )), 4),
                    "mean_change_inside_lung_fraction": round(float(np.mean(
                        [r["signal_inside_lung_fraction"] for r in direction_rows]
                    )), 4),
                    "mean_active_change_outside_lung_fraction": round(float(np.mean(
                        [r["active_outside_lung_fraction"] for r in direction_rows]
                    )), 4),
                }
                for direction in sorted({r["direction"] for r in gan_results})
                for direction_rows in [[
                    r for r in gan_results if r["direction"] == direction
                ]]
            },
            "per_image": gan_results,
        },
    }

    with open(OUTPUT_DIR / "explanation_quality.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExplanation quality evaluation complete.")
    print(f"  Grad-CAM faithfulness: {summary['gradcam']['mean_faithfulness']:.4f}")
    print("  Grad-CAM signal inside lung: "
          f"{summary['gradcam']['mean_cam_signal_inside_lung_fraction']:.4f}")
    if cf_results:
        print(f"  Standard CF SSIM: {summary['standard_counterfactual']['mean_ssim']:.4f}, "
              f"sparsity: {summary['standard_counterfactual']['mean_sparsity']:.4f}, "
              "change inside lung: "
              f"{summary['standard_counterfactual']['mean_change_inside_lung_fraction']:.4f}")
    if guided_results:
        print(f"  Guided CF SSIM: {summary['guided_counterfactual']['mean_ssim']:.4f}, "
              f"sparsity: {summary['guided_counterfactual']['mean_sparsity']:.4f}, "
              "change inside lung: "
              f"{summary['guided_counterfactual']['mean_change_inside_lung_fraction']:.4f}")
    if gan_results:
        print(f"  GAN CF SSIM: {summary['gan_counterfactual']['mean_ssim']:.4f}, "
              f"sparsity: {summary['gan_counterfactual']['mean_sparsity']:.4f}, "
              "change inside lung: "
              f"{summary['gan_counterfactual']['mean_change_inside_lung_fraction']:.4f}")
    print("\nStep 8 outputs")
    print(f"  Output directory:              {OUTPUT_DIR}")
    print(f"  Quality metrics JSON:          {OUTPUT_DIR / 'explanation_quality.json'}")
    print(f"  Grad-CAM fidelity chart:       {OUTPUT_DIR / 'gradcam_fidelity_chart.png'}")
    print(f"  Insertion/deletion curves:     {OUTPUT_DIR / 'insertion_deletion_curves.png'}")
    print(f"  Counterfactual quality chart:  {OUTPUT_DIR / 'counterfactual_quality_chart.png'}")
    print(f"  Anatomical plausibility chart: {OUTPUT_DIR / 'anatomical_plausibility_chart.png'}")
    print(f"\nDone! Run 09_compare_gradcam_counterfactual.py next.")


if __name__ == "__main__":
    main()
