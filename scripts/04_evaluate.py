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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             ConfusionMatrixDisplay)
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DROPOUT = 0.3
NUM_CLASSES = 2

DATA_DIR = Path("data/raw/chest_xray")
RESULTS_DIR = Path("experiments/results")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


# --- Dataset ---
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples, self.labels = [], []
        for label_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = self.root_dir / cls
            if cls_dir.exists():
                for p in sorted(cls_dir.glob("*.jpeg")):
                    self.samples.append(p)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ChestXRayDataset(DATA_DIR, "test", eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)
print(f"Test set: {len(test_dataset)} images")

# --- Load model ---
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, NUM_CLASSES),
)

checkpoint = torch.load("experiments/checkpoints/best_classifier.pt",
                        map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f})")

# --- Inference ---
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# --- Metrics ---
metrics = {
    "accuracy": float(accuracy_score(all_labels, all_preds)),
    "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
    "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
    "f1_score": float(f1_score(all_labels, all_preds, zero_division=0)),
    "auc_roc": float(roc_auc_score(all_labels, all_probs)),
}

print("\n" + "=" * 50)
print("TEST SET RESULTS")
print("=" * 50)
for k, v in metrics.items():
    print(f"  {k:>12}: {v:.4f}")

print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
plt.title("Test Set Confusion Matrix")
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'confusion_matrix.png'}")

# --- ROC curve ---
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'roc_curve.png'}")

# --- Save metrics ---
with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'test_metrics.json'}")

print("\nDone! Run 05_gradcam.py next.")
