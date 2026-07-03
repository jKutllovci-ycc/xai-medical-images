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

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 25
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
NUM_CLASSES = 2
PATIENCE = 5

DATA_DIR = Path("data/raw/chest_xray")
CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_DIR = Path("experiments/results")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# --- Dataset ---
class ChestXRayDataset(Dataset):
    CLASSES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []

        for label_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
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


# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- DataLoaders ---
train_dataset = ChestXRayDataset(DATA_DIR, "train", train_transform)
val_dataset = ChestXRayDataset(DATA_DIR, "val", eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")

# --- Model ---
model = models.resnet18(weights="IMAGENET1K_V1")
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(in_features, NUM_CLASSES),
)
model = model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- Training loop ---
best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
           "val_f1": [], "val_auc": []}

print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # Train
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_dataset)
    train_acc = accuracy_score(all_labels, all_preds)

    # Validate
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    val_loss = running_loss / len(val_dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.0

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)

    elapsed = time.time() - start
    print(f"\nEpoch {epoch}/{EPOCHS} ({elapsed:.1f}s)")
    print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, CHECKPOINT_DIR / "best_classifier.pt")
        print("  -> Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# --- Save history ---
with open(RESULTS_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\nSaved: {RESULTS_DIR / 'training_history.json'}")

# --- Save training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"], label="Val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend()

ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"], label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'training_curves.png'}")
print("\nDone! Run 04_evaluate.py next.")
