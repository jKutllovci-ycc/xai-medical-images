"""
Step 7: Grad-CAM-guided counterfactual explanations.

Usage:
    python scripts/07_gradcam_guided_counterfactual.py
    python scripts/07_gradcam_guided_counterfactual.py --steps 800 --max-per-class 8

What it does:
    - Combines Grad-CAM with counterfactual generation for more precise results
    - Standard counterfactuals (step 06) perturb the entire lung field, which can
      produce diffuse, hard-to-interpret changes spread across the whole image
    - This script first computes a Grad-CAM++ heatmap to find WHERE the model is
      looking, then restricts the counterfactual perturbation to ONLY those regions
    - Result: smaller, more focused perturbations that change only the pixels the
      model actually uses for its decision — much easier to interpret clinically
    - Saves side-by-side panels comparing standard vs guided counterfactuals
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
from pytorch_grad_cam import GradCAMPlusPlus
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
OUTPUT_DIR = Path("outputs/guided_counterfactuals")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
                    "normal_prob": float(prob[0]),
                    "pneumonia_prob": float(prob[1]),
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


def load_display_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
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


def compute_gradcam_mask(model, input_tensor, target_idx, device, cam_threshold=0.3):
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_idx)],
    )[0]

    gradcam_mask = np.where(grayscale_cam >= cam_threshold, 1.0, 0.0).astype(np.float32)
    gradcam_mask = cv2.dilate(gradcam_mask, np.ones((7, 7), np.uint8), iterations=1)
    gradcam_mask = cv2.GaussianBlur(gradcam_mask, (5, 5), 0)
    gradcam_mask = np.clip(gradcam_mask, 0, 1)

    return grayscale_cam, gradcam_mask


def make_guided_mask(lung_mask, gradcam_mask, blend_weight=0.7):
    combined = lung_mask * (blend_weight * gradcam_mask + (1 - blend_weight))
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    return np.clip(combined, 0, 1).astype(np.float32)


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
    best_cf_prob = 0.0
    best_loss = float("inf")

    for step in range(steps):
        optimizer.zero_grad()
        cf_input = input_tensor + delta * mask_tensor
        logits = model(cf_input)
        probs = F.softmax(logits, dim=1)

        target_prob = probs[0, target_class]

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

        current_prob = target_prob.item()
        if current_prob > best_cf_prob:
            best_cf_prob = current_prob
            best_delta = delta.detach().clone()
            best_loss = loss.item()

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


def save_guided_panel(
    row, original_rgb, gradcam_heatmap, lung_mask, guided_mask,
    cf_rgb_standard, cf_rgb_guided,
    diff_standard, diff_guided,
    cf_label_std, cf_conf_std, cf_label_guided, cf_conf_guided,
    output_dir, idx,
):
    true_label = row["true_label"].lower()
    stem = f"{idx:02d}_{true_label}_{Path(row['path']).stem}"

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title(
        f"Original\nTrue: {row['true_label']} | Pred: {row['pred_label']} ({row['confidence']:.3f})",
        fontsize=9,
    )

    axes[0, 1].imshow(gradcam_heatmap, cmap="turbo", vmin=0, vmax=1)
    axes[0, 1].set_title("Grad-CAM++ Heatmap\n(where model looks)", fontsize=9)

    axes[0, 2].imshow(lung_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Lung-field Mask\n(anatomy constraint)", fontsize=9)

    axes[0, 3].imshow(guided_mask, cmap="hot", vmin=0, vmax=1)
    axes[0, 3].set_title("Guided Mask\n(Grad-CAM + lung field)", fontsize=9)

    axes[1, 0].imshow(cf_rgb_standard)
    axes[1, 0].set_title(
        f"Standard CF\n-> {cf_label_std} ({cf_conf_std:.3f})",
        fontsize=9,
    )

    axes[1, 1].imshow(diff_standard)
    axes[1, 1].set_title("Standard Diff\n(changes spread widely)", fontsize=9)

    axes[1, 2].imshow(cf_rgb_guided)
    axes[1, 2].set_title(
        f"Guided CF\n-> {cf_label_guided} ({cf_conf_guided:.3f})",
        fontsize=9,
    )

    axes[1, 3].imshow(diff_guided)
    axes[1, 3].set_title("Guided Diff\n(changes focused on model attention)", fontsize=9)

    for ax in axes.reshape(-1):
        ax.axis("off")

    status = "correct" if row["correct"] else "misclassified"
    fig.suptitle(
        f"{Path(row['path']).name}  |  {status}  |  "
        f"Standard vs Grad-CAM-Guided Counterfactual",
        fontsize=12, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    panel_path = output_dir / f"{stem}_guided_panel.png"
    plt.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    Image.fromarray(np.uint8(cf_rgb_guided * 255)).save(
        output_dir / f"{stem}_guided_cf.png"
    )
    plt.imsave(
        output_dir / f"{stem}_guided_diff.png",
        diff_guided[..., :3] if diff_guided.ndim == 3 else diff_guided,
        cmap="turbo" if diff_guided.ndim == 2 else None,
    )

    return panel_path


def save_summary_grid(results, output_dir):
    successful = [r for r in results if r["guided_cf_success"]]
    if not successful:
        print("  No successful guided counterfactuals for summary grid.")
        return

    n = len(successful)
    fig, axes = plt.subplots(n, 4, figsize=(20, n * 4))
    axes = np.atleast_2d(axes)

    for i, r in enumerate(successful):
        axes[i, 0].imshow(r["original_rgb"])
        axes[i, 0].set_title(
            f"Original: {r['pred_label']} ({r['confidence']:.2f})",
            fontsize=8,
        )

        axes[i, 1].imshow(r["gradcam_heatmap"], cmap="turbo", vmin=0, vmax=1)
        axes[i, 1].set_title("Grad-CAM attention", fontsize=8)

        axes[i, 2].imshow(r["guided_cf_rgb"])
        axes[i, 2].set_title(
            f"Guided CF: {r['guided_cf_pred_label']} ({r['guided_cf_confidence']:.2f})",
            fontsize=8,
        )

        axes[i, 3].imshow(r["guided_diff_overlay"])
        axes[i, 3].set_title("Focused changes", fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(
        "Grad-CAM-Guided Counterfactuals: Precise, Focused Changes",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "guided_cf_summary_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(results, output_dir):
    successful = [r for r in results
                  if r["standard_cf_success"] or r["guided_cf_success"]]
    if not successful:
        return

    n = len(successful)
    fig, axes = plt.subplots(n, 3, figsize=(16, n * 4))
    axes = np.atleast_2d(axes)

    for i, r in enumerate(successful):
        axes[i, 0].imshow(r["original_rgb"])
        axes[i, 0].set_title(
            f"Original: {r['pred_label']} ({r['confidence']:.2f})",
            fontsize=8,
        )

        axes[i, 1].imshow(r["standard_diff_overlay"])
        std_status = "flipped" if r["standard_cf_success"] else "partial"
        axes[i, 1].set_title(
            f"Standard CF diff [{std_status}]\nL1={r['standard_l1']:.4f}",
            fontsize=8,
        )

        axes[i, 2].imshow(r["guided_diff_overlay"])
        g_status = "flipped" if r["guided_cf_success"] else "partial"
        axes[i, 2].set_title(
            f"Guided CF diff [{g_status}]\nL1={r['guided_l1']:.4f} "
            f"({r['l1_reduction_pct']:.0f}% smaller)",
            fontsize=8,
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(
        "Standard vs Guided Counterfactuals: More Precise with Less Change",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "standard_vs_guided_grid.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def write_metadata(results, all_rows, output_dir, args):
    serialisable = []
    for r in results:
        entry = {k: v for k, v in r.items()
                 if k not in {
                     "original_rgb", "gradcam_heatmap", "guided_cf_rgb",
                     "standard_cf_rgb", "guided_diff_overlay", "standard_diff_overlay",
                 }}
        serialisable.append(entry)

    metadata = {
        "cam_threshold": args.cam_threshold,
        "blend_weight": args.blend_weight,
        "steps": args.steps,
        "lr": args.lr,
        "l1_weight": args.l1_weight,
        "tv_weight": args.tv_weight,
        "confidence_threshold": args.confidence_threshold,
        "max_per_class": args.max_per_class,
        "selected_count": len(results),
        "guided_success_count": sum(1 for r in results if r["guided_cf_success"]),
        "standard_success_count": sum(1 for r in results if r["standard_cf_success"]),
        "all_test_count": len(all_rows),
        "avg_l1_reduction_pct": round(float(np.mean([
            r["l1_reduction_pct"] for r in results
            if r["guided_cf_success"] and r["standard_cf_success"]
        ])), 1) if any(r["guided_cf_success"] and r["standard_cf_success"] for r in results) else 0,
        "selected": serialisable,
    }
    with open(output_dir / "guided_cf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    fieldnames = [
        "file", "path", "true_label", "pred_label", "correct", "confidence",
        "standard_cf_pred_label", "standard_cf_confidence", "standard_cf_success", "standard_l1",
        "guided_cf_pred_label", "guided_cf_confidence", "guided_cf_success", "guided_l1",
        "l1_reduction_pct", "guided_mask_coverage",
        "panel_path",
    ]
    with open(output_dir / "guided_cf_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({name: r.get(name) for name in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM-guided counterfactual explanations."
    )
    parser.add_argument("--max-per-class", type=int, default=6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l1-weight", type=float, default=0.008)
    parser.add_argument("--tv-weight", type=float, default=0.003)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--cam-threshold", type=float, default=0.3,
                        help="Grad-CAM activation threshold for the guided mask. "
                             "Lower = wider mask (more pixels allowed to change).")
    parser.add_argument("--blend-weight", type=float, default=0.7,
                        help="How strongly Grad-CAM focuses the mask (0=lung only, 1=CAM only).")
    parser.add_argument("--no-clean", action="store_true")
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
            f"No test images found under {DATA_DIR}. "
            "Run scripts/01_download_data.py first."
        )

    model = build_model(device)
    all_rows = score_test_set(model, dataset, device)
    selected = select_examples(all_rows, args.max_per_class)
    print(f"Selected {len(selected)} examples from {len(all_rows)} test images")
    print(f"CAM threshold: {args.cam_threshold}, blend weight: {args.blend_weight}")

    results = []
    for idx, row in enumerate(tqdm(selected, desc="Generating guided counterfactuals")):
        rgb = load_display_image(row["path"])
        lung_mask = make_lung_field_mask(rgb)
        input_tensor = (
            transform(Image.open(row["path"]).convert("RGB"))
            .unsqueeze(0).to(device)
        )

        source_class = row["pred_idx"]
        target_class = 1 - source_class

        gradcam_heatmap, gradcam_mask = compute_gradcam_mask(
            model, input_tensor, source_class, device,
            cam_threshold=args.cam_threshold,
        )

        guided_mask = make_guided_mask(lung_mask, gradcam_mask, args.blend_weight)

        lung_mask_tensor = (
            torch.from_numpy(lung_mask)
            .unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device)
        )
        guided_mask_tensor = (
            torch.from_numpy(guided_mask)
            .unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device)
        )

        std_delta, std_success, std_prob, _ = generate_counterfactual(
            model, input_tensor, source_class, target_class, lung_mask_tensor,
            device, args.steps, args.lr, args.l1_weight, args.tv_weight,
            args.confidence_threshold,
        )

        guided_delta, guided_success, guided_prob, _ = generate_counterfactual(
            model, input_tensor, source_class, target_class, guided_mask_tensor,
            device, args.steps, args.lr, args.l1_weight, args.tv_weight,
            args.confidence_threshold,
        )

        with torch.no_grad():
            std_cf_tensor = input_tensor + std_delta * lung_mask_tensor
            std_logits = model(std_cf_tensor)
            std_probs = F.softmax(std_logits, dim=1).cpu().numpy()[0]
            std_pred_idx = int(std_probs.argmax())
            std_confidence = float(std_probs[std_pred_idx])
            std_pred_label = CLASS_NAMES[std_pred_idx]

            guided_cf_tensor = input_tensor + guided_delta * guided_mask_tensor
            guided_logits = model(guided_cf_tensor)
            guided_probs = F.softmax(guided_logits, dim=1).cpu().numpy()[0]
            guided_pred_idx = int(guided_probs.argmax())
            guided_confidence = float(guided_probs[guided_pred_idx])
            guided_pred_label = CLASS_NAMES[guided_pred_idx]

        original_rgb = tensor_to_display(input_tensor)
        std_cf_rgb = tensor_to_display(std_cf_tensor)
        guided_cf_rgb = tensor_to_display(guided_cf_tensor)

        std_diff_gray, std_diff_overlay = make_difference_map(original_rgb, std_cf_rgb)
        guided_diff_gray, guided_diff_overlay = make_difference_map(original_rgb, guided_cf_rgb)

        std_l1 = float(np.abs((std_delta * lung_mask_tensor).detach().cpu().numpy()).mean())
        guided_l1 = float(np.abs((guided_delta * guided_mask_tensor).detach().cpu().numpy()).mean())
        l1_reduction = (1 - guided_l1 / max(std_l1, 1e-8)) * 100

        guided_mask_coverage = float(guided_mask.sum()) / max(float(lung_mask.sum()), 1)

        panel_path = save_guided_panel(
            row, original_rgb, gradcam_heatmap, lung_mask, guided_mask,
            std_cf_rgb, guided_cf_rgb,
            std_diff_overlay, guided_diff_overlay,
            std_pred_label, std_confidence,
            guided_pred_label, guided_confidence,
            OUTPUT_DIR, idx,
        )

        result = {
            **row,
            "standard_cf_pred_label": std_pred_label,
            "standard_cf_confidence": round(std_confidence, 4),
            "standard_cf_success": std_success,
            "standard_l1": round(std_l1, 6),
            "guided_cf_pred_label": guided_pred_label,
            "guided_cf_confidence": round(guided_confidence, 4),
            "guided_cf_success": guided_success,
            "guided_l1": round(guided_l1, 6),
            "l1_reduction_pct": round(l1_reduction, 1),
            "guided_mask_coverage": round(guided_mask_coverage, 4),
            "panel_path": str(panel_path),
            "original_rgb": original_rgb,
            "gradcam_heatmap": gradcam_heatmap,
            "guided_cf_rgb": guided_cf_rgb,
            "standard_cf_rgb": std_cf_rgb,
            "guided_diff_overlay": guided_diff_overlay,
            "standard_diff_overlay": std_diff_overlay,
        }
        results.append(result)

        g_status = "FLIPPED" if guided_success else "PARTIAL"
        s_status = "FLIPPED" if std_success else "PARTIAL"
        print(
            f"  [{g_status}] {row['file']}: "
            f"standard L1={std_l1:.4f} [{s_status}], "
            f"guided L1={guided_l1:.4f} [{g_status}], "
            f"reduction={l1_reduction:.0f}%"
        )

    save_summary_grid(results, OUTPUT_DIR)
    save_comparison_grid(results, OUTPUT_DIR)
    write_metadata(results, all_rows, OUTPUT_DIR, args)

    n_guided_success = sum(1 for r in results if r["guided_cf_success"])
    n_std_success = sum(1 for r in results if r["standard_cf_success"])
    n_total = len(results)

    print(f"\nGuided counterfactual generation complete.")
    print(f"  Standard CF success: {n_std_success}/{n_total}")
    print(f"  Guided CF success:   {n_guided_success}/{n_total}")
    if any(r["guided_cf_success"] and r["standard_cf_success"] for r in results):
        avg_reduction = np.mean([
            r["l1_reduction_pct"] for r in results
            if r["guided_cf_success"] and r["standard_cf_success"]
        ])
        print(f"  Avg L1 reduction:    {avg_reduction:.1f}% (guided vs standard)")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Summary: {OUTPUT_DIR / 'guided_cf_summary_grid.png'}")
    print(f"  Comparison: {OUTPUT_DIR / 'standard_vs_guided_grid.png'}")


if __name__ == "__main__":
    main()
