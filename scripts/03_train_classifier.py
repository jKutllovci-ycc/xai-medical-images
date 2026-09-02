"""
Step 3: Train the pneumonia classifier.

Usage:
    python scripts/03_train_classifier.py

What it does:
    - Loads chest X-ray data with augmentation
    - Trains a ResNet-18 classifier (pretrained on ImageNet)
    - Early stopping on validation loss
    - Saves best model to experiments/checkpoints/best_classifier.pt
    - Saves training history to experiments/results/training_history.json
    - Saves training curves plot to experiments/results/training_curves.png
"""

import argparse
import csv
import json
import os
import time
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
BACKBONE_LR = 0.00005
HEAD_LR = 0.0003
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
NUM_CLASSES = 2
PATIENCE = 8
LABEL_SMOOTHING = 0.05
SEED = 42

DATA_DIR = Path("data/raw/chest_xray")
SPLIT_CSV = Path("data/processed/splits/stratified_split.csv")
CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_DIR = Path("experiments/results")
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
    def __init__(self, root_dir, split="train", transform=None, split_csv=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

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
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        return {cls: self.labels.count(idx) for idx, cls in enumerate(CLASS_NAMES)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_transform(preprocess_mode="baseline"):
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_image_preprocessing(img, preprocess_mode)),
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        ], p=0.7),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(preprocess_mode="baseline"):
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_image_preprocessing(img, preprocess_mode)),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_balanced_sampler(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_class_weight_tensor(labels, device):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = counts.sum()
    weights = total / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(labels, preds, probs):
    metrics = {
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", choices=["baseline", "clahe", "clahe_denoise"],
                        default="baseline",
                        help="Image preprocessing mode applied before resize/normalization.")
    parser.add_argument("--raw-splits", action="store_true",
                        help="Use the original Kaggle train/val/test folders instead of the stratified split CSV.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Maximum training epochs (default: {EPOCHS}).")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early-stopping patience (default: {PATIENCE}).")
    parser.add_argument("--min-epochs", type=int, default=15,
                        help="Do not early-stop before this epoch (default: 15).")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    pin_memory = device.type == "cuda"
    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    print(f"Using device: {device}")

    train_transform = build_train_transform(args.preprocess)
    eval_transform = build_eval_transform(args.preprocess)
    split_csv = None if args.raw_splits else SPLIT_CSV
    if split_csv and not split_csv.exists():
        print(f"Stratified split not found at {split_csv}; falling back to raw folders.")
        split_csv = None

    # --- DataLoaders ---
    train_dataset = ChestXRayDataset(DATA_DIR, "train", train_transform, split_csv=split_csv)
    val_dataset = ChestXRayDataset(DATA_DIR, "val", eval_transform, split_csv=split_csv)

    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Preprocessing: {args.preprocess}")
    print(f"Split source: {'stratified CSV' if split_csv else 'raw Kaggle folders'}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit(
            f"No images found under {DATA_DIR}. Run scripts/01_download_data.py first."
        )

    train_sampler = make_balanced_sampler(train_dataset.labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train class counts: {train_dataset.class_counts()}")
    print(f"Val class counts:   {val_dataset.class_counts()}")
    print("Augmentation: resize/crop, mild affine, mild contrast; no horizontal flip")

    # --- Model ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training setup ---
    class_weight_tensor = make_class_weight_tensor(train_dataset.labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
         "lr": BACKBONE_LR},
        {"params": model.fc.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_val_balanced_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_balanced_acc": [], "val_balanced_acc": [],
               "val_precision": [], "val_recall": [], "val_f1": [], "val_auc": []}

    print(
        f"\nTraining for up to {args.epochs} epochs "
        f"(patience={args.patience}, min_epochs={args.min_epochs})...\n"
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # Train
        model.train()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

        train_loss = running_loss / len(train_dataset)
        train_metrics = compute_metrics(all_labels, all_preds, all_probs)

        # Validate
        model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        val_loss = running_loss / len(val_dataset)
        val_metrics = compute_metrics(all_labels, all_preds, all_probs)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["train_balanced_acc"].append(train_metrics["balanced_acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_balanced_acc"].append(val_metrics["balanced_acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        elapsed = time.time() - start
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(
            f"  Train - Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} | "
            f"BalAcc: {train_metrics['balanced_acc']:.4f}"
        )
        print(
            f"  Val   - Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} | "
            f"BalAcc: {val_metrics['balanced_acc']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}"
        )

        improved = (
            val_metrics["balanced_acc"] > best_val_balanced_acc + 1e-4 or
            (abs(val_metrics["balanced_acc"] - best_val_balanced_acc) <= 1e-4 and
             val_loss < best_val_loss)
        )
        if improved:
            best_val_loss = val_loss
            best_val_balanced_acc = val_metrics["balanced_acc"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "class_names": CLASS_NAMES,
                "class_counts": train_dataset.class_counts(),
                "recommended_class_weights": class_weight_tensor.detach().cpu().tolist(),
                "config": {
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "epochs": args.epochs,
                    "min_epochs": args.min_epochs,
                    "backbone_lr": BACKBONE_LR,
                    "head_lr": HEAD_LR,
                    "weight_decay": WEIGHT_DECAY,
                    "dropout": DROPOUT,
                    "label_smoothing": LABEL_SMOOTHING,
                    "balanced_sampler": True,
                    "horizontal_flip": False,
                    "preprocess_mode": args.preprocess,
                    "split_csv": str(split_csv) if split_csv else "",
                    "stratified_split": bool(split_csv),
                    "seed": SEED,
                },
            }, CHECKPOINT_DIR / "best_classifier.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            can_stop = epoch >= args.min_epochs
            if patience_counter >= args.patience and can_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            if patience_counter >= args.patience and not can_stop:
                print(
                    f"  -> No improvement for {patience_counter} epoch(s), "
                    f"but continuing until min_epochs={args.min_epochs}"
                )

    # --- Save history ---
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'training_history.json'}")

    # --- Save training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Training images")
    ax1.plot(history["val_loss"], label="Validation images")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error/loss")
    ax1.set_title("Learning error over time\nLower means the model is fitting better")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Training accuracy")
    ax2.plot(history["val_acc"], label="Validation accuracy")
    ax2.plot(history["val_balanced_acc"], label="Validation balanced accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Correct predictions over time\nBalanced accuracy checks both classes fairly")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'training_curves.png'}")

    print("\nStep 3 outputs")
    print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"  Best checkpoint:      {CHECKPOINT_DIR / 'best_classifier.pt'}")
    print(f"  Results directory:    {RESULTS_DIR}")
    print(f"  Training history:     {RESULTS_DIR / 'training_history.json'}")
    print(f"  Training curves:      {RESULTS_DIR / 'training_curves.png'}")
    print("\nDone! Run 04_evaluate.py next.")


if __name__ == "__main__":
    freeze_support()
    main()
