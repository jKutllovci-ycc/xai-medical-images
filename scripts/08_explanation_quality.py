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
    - These metrics answer: "Is the model actually using the regions the explanation
      highlights?" and "Are the counterfactual changes minimal and realistic?"
    - Saves metrics to JSON and generates comparison bar charts
"""

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
CHECKPOINT_PATH = Path("experiments/checkpoints/best_classifier.pt")
RESULTS_DIR = Path("experiments/results")
GRADCAM_DIR = Path("outputs/gradcam")
CF_DIR = Path("outputs/counterfactuals")
GUIDED_CF_DIR = Path("outputs/guided_counterfactuals")
OUTPUT_DIR = Path("outputs/explanation_quality")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def trapezoid_auc(values):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values)
    return np.trapz(values)


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


def compute_counterfactual_metrics(original_rgb, cf_rgb):
    ssim_val = ssim(
        original_rgb, cf_rgb,
        channel_axis=2,
        data_range=1.0,
    )
    l1_dist = float(np.abs(original_rgb - cf_rgb).mean())
    l2_dist = float(np.sqrt(np.mean((original_rgb - cf_rgb) ** 2)))

    diff = np.abs(original_rgb - cf_rgb).mean(axis=2)
    changed_pixels = float((diff > 0.02).mean())

    return {
        "ssim": round(float(ssim_val), 4),
        "l1_distance": round(l1_dist, 6),
        "l2_distance": round(l2_dist, 6),
        "changed_pixel_fraction": round(changed_pixels, 4),
        "sparsity": round(1.0 - changed_pixels, 4),
    }


def evaluate_gradcam(model, selected, transform, device):
    print("\nEvaluating Grad-CAM quality...")
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    results = []

    for row in tqdm(selected, desc="Grad-CAM fidelity"):
        input_tensor = transform(
            Image.open(row["path"]).convert("RGB")
        ).unsqueeze(0).to(device)

        target_idx = CLASS_NAMES.index("PNEUMONIA")
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_idx)],
        )[0]

        fidelity = insertion_deletion_test(
            model, input_tensor, grayscale_cam, target_idx, device,
        )

        cam_sparsity = float((grayscale_cam < 0.1).mean())
        cam_peak = float(grayscale_cam.max())
        cam_coverage = float((grayscale_cam >= 0.3).mean())

        results.append({
            "file": row["file"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            **fidelity,
            "cam_sparsity": round(cam_sparsity, 4),
            "cam_peak": round(cam_peak, 4),
            "cam_coverage_30pct": round(cam_coverage, 4),
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
    for entry in meta.get("selected", []):
        path = entry.get("path", "")
        file_stem = Path(path).stem
        cf_candidates = list(cf_dir.glob(f"*{file_stem}*_counterfactual.png"))
        if not cf_candidates or not Path(path).exists():
            continue

        original = np.asarray(
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0
        cf_img = np.asarray(
            Image.open(cf_candidates[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0

        metrics = compute_counterfactual_metrics(original, cf_img)
        results.append({
            "file": entry.get("file", ""),
            "true_label": entry.get("true_label", ""),
            "pred_label": entry.get("pred_label", ""),
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
    for entry in meta.get("selected", []):
        path = entry.get("path", "")
        file_stem = Path(path).stem
        cf_candidates = list(guided_dir.glob(f"*{file_stem}*_guided_cf.png"))
        if not cf_candidates or not Path(path).exists():
            continue

        original = np.asarray(
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0
        cf_img = np.asarray(
            Image.open(cf_candidates[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0

        metrics = compute_counterfactual_metrics(original, cf_img)
        results.append({
            "file": entry.get("file", ""),
            "true_label": entry.get("true_label", ""),
            "pred_label": entry.get("pred_label", ""),
            "guided_cf_pred_label": entry.get("guided_cf_pred_label", ""),
            "guided_cf_success": entry.get("guided_cf_success", False),
            "guided_cf_confidence": entry.get("guided_cf_confidence", 0),
            "standard_l1": entry.get("standard_l1", 0),
            "guided_l1": entry.get("guided_l1", 0),
            "l1_reduction_pct": entry.get("l1_reduction_pct", 0),
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
    axes[0].bar(x - w / 2, ins_aucs, w, label="Insertion AUC", color="#22c55e")
    axes[0].bar(x + w / 2, del_aucs, w, label="Deletion AUC", color="#ef4444")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Grad-CAM Insertion vs Deletion AUC\n(higher insertion + lower deletion = more faithful)")
    axes[0].legend()

    colors = ["#22c55e" if f > 0.3 else "#f59e0b" if f > 0.1 else "#ef4444"
              for f in faithfulness]
    axes[1].bar(x, faithfulness, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Faithfulness (Ins - Del)")
    axes[1].set_title("Grad-CAM Faithfulness Score\n(green=good, red=poor)")
    axes[1].axhline(y=0.3, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_fidelity_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_counterfactual_quality_chart(cf_results, guided_results, output_dir):
    if not cf_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cf_ssim = [r["ssim"] for r in cf_results]
    cf_sparsity = [r["sparsity"] for r in cf_results]
    cf_l1 = [r["l1_distance"] for r in cf_results]

    labels = ["Standard CF"]
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

    colors = ["#3b82f6", "#22c55e"][:len(labels)]

    axes[0].bar(labels, ssim_means, color=colors)
    axes[0].set_title("SSIM (higher = more similar to original)\nCloser to 1.0 = more realistic")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(ssim_means):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[1].bar(labels, sparsity_means, color=colors)
    axes[1].set_title("Sparsity (higher = fewer pixels changed)\nMore sparse = more focused explanation")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(sparsity_means):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[2].bar(labels, l1_means, color=colors)
    axes[2].set_title("L1 Distance (lower = smaller perturbation)\nSmaller = more minimal change")
    for i, v in enumerate(l1_means):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "counterfactual_quality_chart.png", dpi=150,
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
        axes[i].plot(steps_x, r["insertion_curve"], "g-", linewidth=2, label="Insertion")
        axes[i].plot(steps_x, r["deletion_curve"], "r-", linewidth=2, label="Deletion")
        axes[i].fill_between(steps_x, r["insertion_curve"], r["deletion_curve"],
                             alpha=0.15, color="blue")
        axes[i].set_xlabel("Fraction of pixels revealed/removed")
        axes[i].set_ylabel("Target class probability")
        status = "OK" if r["correct"] else "MISS"
        axes[i].set_title(
            f"{status} {r['file'][:25]}\nFaith={r['faithfulness']:.3f}",
            fontsize=9,
        )
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(-0.05, 1.05)

    plt.suptitle(
        "Insertion/Deletion Curves — Does Grad-CAM highlight what the model actually uses?",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "insertion_deletion_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    selected = select_examples(all_rows, max_per_class=6)

    gradcam_results = evaluate_gradcam(model, selected, transform, device)

    print("\nEvaluating standard counterfactual quality...")
    cf_results = evaluate_counterfactuals(CF_DIR, "Standard CF")

    print("Evaluating guided counterfactual quality...")
    guided_results = evaluate_guided_counterfactuals(GUIDED_CF_DIR)

    save_fidelity_chart(gradcam_results, OUTPUT_DIR)
    save_insertion_deletion_curves(gradcam_results, OUTPUT_DIR)
    save_counterfactual_quality_chart(cf_results, guided_results, OUTPUT_DIR)

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
            "success_rate": round(
                sum(1 for r in guided_results if r.get("guided_cf_success", False)) /
                max(len(guided_results), 1), 4
            ) if guided_results else 0,
            "per_image": guided_results,
        },
    }

    with open(OUTPUT_DIR / "explanation_quality.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExplanation quality evaluation complete.")
    print(f"  Grad-CAM faithfulness: {summary['gradcam']['mean_faithfulness']:.4f}")
    if cf_results:
        print(f"  Standard CF SSIM: {summary['standard_counterfactual']['mean_ssim']:.4f}, "
              f"sparsity: {summary['standard_counterfactual']['mean_sparsity']:.4f}")
    if guided_results:
        print(f"  Guided CF SSIM: {summary['guided_counterfactual']['mean_ssim']:.4f}, "
              f"sparsity: {summary['guided_counterfactual']['mean_sparsity']:.4f}")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"\nDone! Run 09_compare_gradcam_counterfactual.py next.")


if __name__ == "__main__":
    main()
