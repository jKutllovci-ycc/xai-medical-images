"""
Step 5: Generate Grad-CAM heatmap explanations.

Usage:
    python scripts/05_gradcam.py

What it does:
    - Loads the trained classifier
    - Generates Grad-CAM heatmaps for selected test images
    - Saves overlays to outputs/gradcam/
    - Creates a summary grid for the notebook
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMG_SIZE = 224
DROPOUT = 0.3
NUM_CLASSES = 2

OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/raw/chest_xray")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# --- Collect sample images: 5 NORMAL + 5 PNEUMONIA from test set ---
sample_paths = []
for cls in CLASS_NAMES:
    cls_dir = DATA_DIR / "test" / cls
    sample_paths.extend([(p, cls) for p in sorted(cls_dir.glob("*.jpeg"))[:5]])

print(f"Generating Grad-CAM for {len(sample_paths)} images...\n")

for idx, (img_path, true_label) in enumerate(sample_paths):
    raw_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_array = np.array(raw_img) / 255.0

    tensor = eval_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_idx = model(tensor).argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=tensor,
                        targets=[ClassifierOutputTarget(pred_idx)])[0]

    overlay = show_cam_on_image(rgb_array.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPred: {CLASS_NAMES[pred_idx]}", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"gradcam_{idx:02d}_{true_label.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{idx+1}/{len(sample_paths)}] {img_path.name} — True: {true_label}, Pred: {CLASS_NAMES[pred_idx]} -> {save_path.name}")

print(f"\nAll Grad-CAM outputs saved to {OUTPUT_DIR}/")
print("Done! Open the notebook to view results.")
