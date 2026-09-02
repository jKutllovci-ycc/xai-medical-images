"""
Step 4: Evaluate the trained classifier on the test set.

Usage:
    python scripts/04_evaluate.py

What it does:
    - Loads the best checkpoint
    - Runs inference on the test set
    - Computes accuracy, precision, recall, F1, AUC-ROC
    - Saves confusion matrix and ROC curve plots
    - Saves full metrics to experiments/results/test_metrics.json
"""

import json
import os
import argparse
from multiprocessing import freeze_support
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DROPOUT = 0.3
NUM_CLASSES = 2

DATA_DIR = Path("data/raw/chest_xray")
SPLIT_CSV = Path("data/processed/splits/stratified_split.csv")
RESULTS_DIR = Path("experiments/results")
CHECKPOINT_PATH = Path("experiments/checkpoints/best_classifier.pt")
DECISION_THRESHOLD_PATH = RESULTS_DIR / "decision_threshold.json"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


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
            for label_idx, cls in enumerate(CLASS_NAMES):
                cls_dir = self.root_dir / split / cls
                if cls_dir.exists():
                    for p in sorted(cls_dir.glob("*.jpeg")):
                        self.samples.append(p)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_eval_transform(preprocess_mode="baseline"):
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_image_preprocessing(img, preprocess_mode)),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_preprocess_mode_from_checkpoint(checkpoint_path, fallback="baseline"):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return fallback
    return checkpoint.get("config", {}).get("preprocess_mode", fallback)


def build_model(device, checkpoint_path):
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_with_threshold(pneumonia_probs, threshold):
    return (np.asarray(pneumonia_probs) >= threshold).astype(int)


def balanced_accuracy_from_predictions(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    specificity = cm[0][0] / max(cm[0].sum(), 1)
    sensitivity = cm[1][1] / max(cm[1].sum(), 1)
    return float((specificity + sensitivity) / 2), float(specificity), float(sensitivity)


def collect_probs(model, loader, device, desc):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def choose_threshold(labels, pneumonia_probs, mode="specificity", target_specificity=0.99):
    thresholds = np.linspace(0.05, 0.95, 181)
    scored = []
    for threshold in thresholds:
        preds = predict_with_threshold(pneumonia_probs, threshold)
        bal_acc, specificity, sensitivity = balanced_accuracy_from_predictions(labels, preds)
        scored.append({
            "threshold": float(threshold),
            "balanced_accuracy": bal_acc,
            "specificity": specificity,
            "recall": sensitivity,
        })

    if mode == "balanced":
        candidates = scored
    else:
        candidates = [r for r in scored if r["specificity"] >= target_specificity]
        if not candidates:
            candidates = scored

    best = max(candidates, key=lambda r: (r["balanced_accuracy"], r["recall"]))
    best["mode"] = mode
    best["target_specificity"] = float(target_specificity)
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold-mode",
        choices=["specificity", "balanced"],
        default="specificity",
        help=(
            "specificity protects NORMAL cases by requiring stronger pneumonia evidence; "
            "balanced maximises validation balanced accuracy."
        ),
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=0.99,
        help="Validation NORMAL detection target used when --threshold-mode=specificity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    preprocess_mode = load_preprocess_mode_from_checkpoint(CHECKPOINT_PATH, fallback="baseline")
    eval_transform = build_eval_transform(preprocess_mode)
    num_workers = NUM_WORKERS if device.type == "cuda" else 0

    split_csv = SPLIT_CSV if SPLIT_CSV.exists() else None
    val_dataset = ChestXRayDataset(DATA_DIR, "val", eval_transform, split_csv=split_csv)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=num_workers)
    test_dataset = ChestXRayDataset(DATA_DIR, "test", eval_transform, split_csv=split_csv)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=num_workers)
    print(f"Test set: {len(test_dataset)} images")
    print(f"Preprocessing: {preprocess_mode}")

    if len(test_dataset) == 0:
        raise SystemExit(
            f"No test images found under {DATA_DIR}. Run scripts/01_download_data.py first."
        )

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"Missing checkpoint: {CHECKPOINT_PATH}. Run scripts/03_train_classifier.py first."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = build_model(device, CHECKPOINT_PATH)
    model.eval()
    print(
        f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss={checkpoint.get('val_loss', '?'):.4f})"
    )

    # --- Calibrate decision threshold on validation data ---
    val_labels, val_probs = collect_probs(model, val_loader, device, "Calibrating threshold")
    threshold_info = choose_threshold(
        val_labels,
        val_probs,
        mode=args.threshold_mode,
        target_specificity=args.target_specificity,
    )
    decision_threshold = threshold_info["threshold"]
    with open(DECISION_THRESHOLD_PATH, "w") as f:
        json.dump({
            "decision_threshold": decision_threshold,
            "chosen_on": "validation split",
            "threshold_mode": args.threshold_mode,
            "target_specificity": args.target_specificity,
            "reason": (
                "Requires stronger pneumonia evidence so NORMAL chest X-rays are less often called PNEUMONIA."
                if args.threshold_mode == "specificity"
                else "Maximises balanced accuracy so NORMAL and PNEUMONIA are treated more fairly."
            ),
            "validation_balanced_accuracy": threshold_info["balanced_accuracy"],
            "validation_specificity": threshold_info["specificity"],
            "validation_recall": threshold_info["recall"],
            "preprocess_mode": preprocess_mode,
            "checkpoint_epoch": checkpoint.get("epoch"),
        }, f, indent=2)
    print(
        f"Decision threshold: {decision_threshold:.3f} "
        f"(validation balanced accuracy={threshold_info['balanced_accuracy']:.4f})"
    )

    # --- Inference ---
    all_labels, all_probs = collect_probs(model, test_loader, device, "Testing")
    all_preds = predict_with_threshold(all_probs, decision_threshold)

    # --- Metrics ---
    cm = confusion_matrix(all_labels, all_preds)
    specificity = cm[0][0] / max(cm[0].sum(), 1)
    sensitivity = cm[1][1] / max(cm[1].sum(), 1)
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1_score": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(all_labels, all_probs)),
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + sensitivity) / 2),
        "preprocess_mode": preprocess_mode,
        "decision_threshold": float(decision_threshold),
    }

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:>18}: {v:.4f}")
        else:
            print(f"  {k:>18}: {v}")

    print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    plt.title(
        "Test results: true label vs model prediction\n"
        "Diagonal boxes are correct predictions; off-diagonal boxes are mistakes"
    )
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'confusion_matrix.png'}")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"Overall separation score (AUC) = {roc_auc:.4f}",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Correct pneumonia detection rate")
    ax.set_title(
        "How well the model separates NORMAL from PNEUMONIA\n"
        "Closer to the top-left corner is better"
    )
    ax.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'roc_curve.png'}")

    # --- Save metrics ---
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'test_metrics.json'}")

    print("\nStep 4 outputs")
    print(f"  Output directory: {RESULTS_DIR}")
    print(f"  Test metrics:     {RESULTS_DIR / 'test_metrics.json'}")
    print(f"  Confusion matrix: {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"  ROC curve:        {RESULTS_DIR / 'roc_curve.png'}")
    print(f"  Decision threshold:{DECISION_THRESHOLD_PATH}")

    print("\nDone! Run 05_gradcam.py next.")


if __name__ == "__main__":
    freeze_support()
    main()
