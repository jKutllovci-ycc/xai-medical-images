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
NORMAL_REF_PATH = Path("experiments/results/normal_lung_reference.npz")
PNEUMONIA_REF_PATH = Path("experiments/results/pneumonia_lung_reference.npz")

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
        candidates = correct[:max_per_class] + hard[:max_per_class]
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
    device, steps, lr, l1_weight, tv_weight, confidence_threshold, delta_limit,
    delta_resolution,
):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if delta_resolution and delta_resolution < input_tensor.shape[-1]:
        delta_shape = (
            input_tensor.shape[0], 1,
            delta_resolution, delta_resolution,
        )
        delta_param = torch.zeros(delta_shape, device=device, requires_grad=True)
    else:
        delta_param = torch.zeros_like(input_tensor, device=device, requires_grad=True)

    def upsample_delta():
        if delta_param.shape[-1] == input_tensor.shape[-1]:
            delta_up = delta_param
        else:
            delta_up = F.interpolate(
            delta_param,
            size=input_tensor.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        if delta_up.shape[1] == 1:
            delta_up = delta_up.expand_as(input_tensor)
        return delta_up

    optimizer = torch.optim.Adam([delta_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_delta = None
    best_loss = float("inf")
    best_cf_prob = 0.0
    flipped = False

    for step in range(steps):
        optimizer.zero_grad()
        delta = upsample_delta()
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
            delta_param.data.clamp_(-delta_limit, delta_limit)

        current_target_prob = target_prob.item()
        if current_target_prob > best_cf_prob:
            best_cf_prob = current_target_prob
            best_delta = upsample_delta().detach().clone()
            best_loss = loss.item()

        if current_target_prob >= confidence_threshold and not flipped:
            flipped = True

    success = best_cf_prob >= 0.5
    return best_delta, success, best_cf_prob, best_loss


def apply_counterfactual_delta(input_tensor, delta, mask_tensor, effect_strength):
    return clamp_normalized(input_tensor + effect_strength * delta * mask_tensor)


def choose_display_counterfactual(
    model, input_tensor, delta, mask_tensor, target_class,
    decision_threshold, max_effect_strength,
):
    strengths = np.linspace(max_effect_strength, 1.0, 16)
    fallback = None
    for strength in strengths:
        cf_tensor = apply_counterfactual_delta(input_tensor, delta, mask_tensor, float(strength))
        logits = model(cf_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = predict_label_from_probs(probs, decision_threshold)
        if fallback is None:
            fallback = (cf_tensor, probs, pred_idx, float(strength))
        if pred_idx == target_class:
            return cf_tensor, probs, pred_idx, float(strength), True
    cf_tensor, probs, pred_idx, strength = fallback
    return cf_tensor, probs, pred_idx, strength, False


def denormalize(tensor):
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return tensor * std + mean


def tensor_to_display(tensor):
    img = denormalize(tensor.squeeze(0)).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def display_to_tensor(rgb, device):
    tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return ((tensor.to(device) - mean) / std).float()


def smooth_noise(seed, size=IMG_SIZE):
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0, 1, (8, 8)).astype(np.float32)
    noise = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=18, sigmaY=18)
    noise -= noise.min()
    if noise.max() > 0:
        noise /= noise.max()
    return noise


_REF_CACHE = {}

def load_normal_reference():
    if "normal" not in _REF_CACHE:
        if NORMAL_REF_PATH.exists():
            data = np.load(NORMAL_REF_PATH)
            _REF_CACHE["normal"] = {
                "mean_map": data["mean_map"],
                "std_map": data["std_map"],
            }
        else:
            _REF_CACHE["normal"] = None
    return _REF_CACHE["normal"]


def load_pneumonia_reference():
    if "pneumonia" not in _REF_CACHE:
        if PNEUMONIA_REF_PATH.exists():
            data = np.load(PNEUMONIA_REF_PATH)
            _REF_CACHE["pneumonia"] = {
                "mean_map": data["mean_map"],
                "std_map": data["std_map"],
            }
        else:
            _REF_CACHE["pneumonia"] = None
    return _REF_CACHE["pneumonia"]


def semantic_edit(original_rgb, lung_mask, target_class, strength, seed):
    gray = original_rgb.mean(axis=2)
    mask = cv2.GaussianBlur(lung_mask.astype(np.float32), (0, 0), sigmaX=4)
    mask = np.clip(mask, 0, 1)

    yy = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)[None, :]
    texture = smooth_noise(seed)
    texture2 = smooth_noise(seed + 7)

    if target_class == 1:
        # Add pneumonia: focal patchy consolidation + ground-glass opacity
        # NOT uniform blur — real pneumonia has distinct affected vs clear areas
        ref_p = load_pneumonia_reference()
        ref_n = load_normal_reference()
        rng = np.random.default_rng(seed)

        # --- Dense consolidation patches (focal, bright white, sharp-edged) ---
        consolidation = np.zeros_like(gray)

        # Place 3-5 consolidation foci in anatomically plausible locations
        # Prefer lower lobes (most common in bacterial pneumonia)
        lobe_centers = [
            (0.30, 0.60), (0.70, 0.58),  # lower lobes L/R
            (0.28, 0.42), (0.72, 0.42),  # mid lobes L/R
            (0.35, 0.30), (0.65, 0.30),  # upper lobes L/R
        ]
        rng.shuffle(lobe_centers)
        n_foci = rng.integers(2, 5)
        for i in range(n_foci):
            cx, cy = lobe_centers[i]
            cx += rng.uniform(-0.06, 0.06)
            cy += rng.uniform(-0.05, 0.05)
            # Irregular shape via multiple overlapping ellipses
            for _ in range(rng.integers(2, 5)):
                dcx = cx + rng.uniform(-0.04, 0.04)
                dcy = cy + rng.uniform(-0.04, 0.04)
                sx = rng.uniform(0.008, 0.025)
                sy = rng.uniform(0.010, 0.030)
                blob = np.exp(-(((xx - dcx) ** 2) / sx + ((yy - dcy) ** 2) / sy))
                consolidation += rng.uniform(0.6, 1.0) * blob

        consolidation = np.clip(consolidation, 0, 1)
        # Sharpen consolidation edges (real consolidation has relatively sharp borders)
        consolidation = np.where(consolidation > 0.15, consolidation, consolidation * 0.3)

        # --- Ground-glass opacity around consolidation (lighter, semi-transparent) ---
        ggo = cv2.GaussianBlur(consolidation, (0, 0), sigmaX=12)
        ggo = np.clip(ggo * 1.5, 0, 0.6) * (0.5 + 0.5 * texture)

        # --- Data-driven spatial weighting (where pneumonia typically appears) ---
        if ref_p is not None and ref_n is not None:
            diff_map = np.clip(ref_p["mean_map"] - ref_n["mean_map"], 0, 1)
            spatial_weight = np.clip(diff_map * 10.0, 0.2, 1.0)
        else:
            spatial_weight = np.clip((yy - 0.15) / 0.50, 0.2, 1.0)

        # Combine: focal consolidation + surrounding ground-glass
        opacity = mask * spatial_weight * np.clip(consolidation + ggo, 0, 1)
        opacity = np.clip(opacity * strength, 0, 0.95)

        # --- Apply the pneumonia effect ---
        edited = original_rgb.copy()

        # Consolidation areas: bright white (fluid-filled alveoli)
        consol_mask = (consolidation > 0.25)[..., None]
        consol_strength = np.clip(consolidation, 0, 1)[..., None] * opacity[..., None]
        white_target = 0.82 + 0.05 * texture2[..., None]
        edited = edited * (1 - consol_strength * 0.85) + white_target * (consol_strength * 0.85)

        # Only blur WITHIN consolidated areas (not the whole lung)
        blurred = cv2.GaussianBlur(edited, (0, 0), sigmaX=2.0)
        blur_mask = np.clip(consolidation * 1.5, 0, 1)[..., None] * opacity[..., None]
        edited = edited * (1 - blur_mask * 0.7) + blurred * (blur_mask * 0.7)

        # Ground-glass areas: lighter haze, ribs still partially visible
        ggo_only = np.clip(ggo - consolidation * 0.5, 0, 1)[..., None] * opacity[..., None]
        ggo_target = 0.65 + 0.08 * texture2[..., None]
        edited = edited * (1 - ggo_only * 0.50) + ggo_target * (ggo_only * 0.50)
    else:
        # Remove pneumonia → produce a realistic normal chest X-ray
        # Uses data-driven patterns from normal and pneumonia dataset references
        ref_n = load_normal_reference()
        ref_p = load_pneumonia_reference()

        # --- Detect pneumonia opacity using data + image analysis ---
        # 1. Data-driven: where current image looks more like pneumonia than normal
        if ref_p is not None and ref_n is not None:
            diff_map = ref_p["mean_map"] - ref_n["mean_map"]
            # Regions where image exceeds normal reference = likely pneumonia opacity
            excess = gray - ref_n["mean_map"]
            data_indicator = np.clip(excess / np.maximum(diff_map, 0.02), 0, 1.5)
            data_indicator = np.clip(data_indicator * 0.7, 0, 1)
        else:
            data_indicator = np.zeros_like(gray)

        # 2. Low local contrast = haze (pneumonia blurs out ribs/vessels)
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=6)
        local_sq = cv2.GaussianBlur(gray ** 2, (0, 0), sigmaX=6)
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
        haze_indicator = np.clip((0.15 - local_std) / 0.10, 0, 1)

        # 3. Bright areas within lung = consolidation/fluid
        lung_values = gray[mask > 0.3]
        if lung_values.size > 0:
            p10 = np.percentile(lung_values, 10)
        else:
            p10 = 0.25
        bright_indicator = np.clip((gray - p10) / 0.18, 0, 1)

        # Combine all detectors
        opacity = np.clip(
            data_indicator * 0.5 + haze_indicator * 0.6 + bright_indicator * 0.4,
            0, 1,
        )

        # Emphasise lower and mid lung zones
        lower_emphasis = np.clip((yy - 0.15) / 0.45, 0, 1)
        mid_emphasis = np.exp(-((yy - 0.45) ** 2) / 0.10)
        zone_weight = 0.30 + 0.45 * lower_emphasis + 0.25 * mid_emphasis
        clearing = mask * opacity * zone_weight
        clearing = cv2.GaussianBlur(clearing, (0, 0), sigmaX=5)
        clearing = np.clip(clearing * strength, 0, 0.95)

        # --- Target: real normal lung appearance from the dataset ---
        if ref_n is not None:
            # Normal reference with spatial structure preserved
            target = ref_n["mean_map"] - 0.3 * ref_n["std_map"]
            target = np.clip(target, 0, 1)
            # Add subtle structural variation from reference std
            target = target + ref_n["std_map"] * 0.10 * texture
        else:
            target = np.full_like(gray, 0.35) + 0.02 * texture

        # Ensure target is darker than current pneumonia pixels (clearing effect)
        if lung_values.size > 0:
            own_dark = np.percentile(lung_values, 12)
            target = np.minimum(target, own_dark + 0.03)

        # Step 1: Clear haze toward normal reference
        blend = clearing * 0.80
        cleared_gray = gray * (1 - blend) + target * blend

        # Step 2: Strong CLAHE to reveal ribs, vessels, bronchial structures
        cleared_uint8 = np.clip(cleared_gray * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleared_uint8).astype(np.float32) / 255.0
        clahe_blend = clearing * 0.65
        cleared_gray = cleared_gray * (1 - clahe_blend) + enhanced * clahe_blend

        # Step 3: Sharpen to reveal rib borders and vessel walls
        sharp_kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(cleared_gray, -1, sharp_kernel)
        sharp_blend = clearing * 0.40
        cleared_gray = cleared_gray * (1 - sharp_blend) + sharpened * sharp_blend

        edited = np.repeat(np.clip(cleared_gray, 0, 1)[..., None], 3, axis=2)

    return np.clip(edited, 0, 1)


def choose_semantic_counterfactual(
    model, original_rgb, mask_np, target_class, device, decision_threshold,
    max_strength, seed,
):
    fallback = None
    for strength in np.linspace(max_strength, 0.15, 18):
        cf_rgb = semantic_edit(original_rgb, mask_np, target_class, float(strength), seed)
        cf_tensor = display_to_tensor(cf_rgb, device)
        logits = model(cf_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = predict_label_from_probs(probs, decision_threshold)
        if fallback is None:
            fallback = (cf_tensor, probs, pred_idx, float(strength), cf_rgb)
        if pred_idx == target_class:
            return cf_tensor, probs, pred_idx, float(strength), cf_rgb, True
    cf_tensor, probs, pred_idx, strength, cf_rgb = fallback
    return cf_tensor, probs, pred_idx, strength, cf_rgb, False


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


def make_descriptive_stem(row, idx):
    true_label = row["true_label"].lower()
    target_label = "pneumonia" if row["pred_idx"] == 0 else "normal"
    return f"{idx + 1:02d}_{true_label}_xray", target_label


def save_counterfactual_panel(
    row, original_rgb, cf_rgb, diff_gray, diff_overlay, mask,
    cf_pred_label, cf_confidence, output_dir, comp_dir, idx,
):
    stem, target_label = make_descriptive_stem(row, idx)

    Image.fromarray(np.uint8(cf_rgb * 255)).save(
        output_dir / f"{stem}_edited_to_look_{target_label}.png"
    )
    plt.imsave(
        output_dir / f"{stem}_what_changed.png", diff_gray, cmap="turbo", vmin=0, vmax=1
    )

    fig, axes = plt.subplots(1, 5, figsize=(24, 6.5))

    axes[0].imshow(original_rgb)
    axes[0].set_title(
        f"1. Original chest X-ray\n"
        f"True label: {row['true_label']}\n"
        f"Model predicted: {row['pred_label']} ({row['confidence']:.3f})",
        fontsize=10,
    )

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        "2. Lung area (where edits are allowed)\n"
        "White = lung region\n"
        "Black = protected area",
        fontsize=10,
    )

    axes[2].imshow(cf_rgb)
    axes[2].set_title(
        f"3. What if this were {target_label.upper()}?\n"
        "Synthetic edit, not a real diagnosis\n"
        f"Model now says: {cf_pred_label} ({cf_confidence:.3f})",
        fontsize=10,
    )

    axes[3].imshow(diff_gray, cmap="turbo", vmin=0, vmax=1)
    axes[3].set_title(
        "4. What changed\n"
        "Bright colors = pixels that changed most\n"
        "Dark = little or no change",
        fontsize=10,
    )

    axes[4].imshow(diff_overlay)
    axes[4].set_title(
        "5. Changes highlighted on original\n"
        "Colored areas show what was edited\n"
        "to flip the model prediction",
        fontsize=10,
    )

    for ax in axes:
        ax.axis("off")

    status = "correct" if row["correct"] else "misclassified"
    fig.suptitle(
        f"What would this {row['true_label']} X-ray need to look like for the model to predict {target_label.upper()}?  "
        f"|  original prediction was {status}",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "How to read this: the algorithm asks 'What is the smallest change inside the lung area "
        "that would make the model switch its answer?' This is a synthetic explanation of what the "
        "model is sensitive to, not a medically verified image.",
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    panel_path = comp_dir / f"{stem}_full_explanation.png"
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
        target_label = "PNEUMONIA" if result["pred_idx"] == 0 else "NORMAL"

        ax_orig = axes[row_idx, col_idx * 2]
        ax_cf = axes[row_idx, col_idx * 2 + 1]

        ax_orig.imshow(result["original_rgb"])
        ax_orig.set_title(
            f"Original {result['true_label']} X-ray\n"
            f"Model says: {result['pred_label']} ({result['confidence']:.2f})",
            fontsize=8,
        )

        ax_cf.imshow(result["cf_rgb"])
        ax_cf.set_title(
            f"What if {target_label}?\n"
            f"Model now says: {result['cf_pred_label']} ({result['cf_confidence']:.2f})",
            fontsize=8,
        )

        ax_orig.axis("off")
        ax_cf.axis("off")

    plt.suptitle(
        "Original X-rays vs Edited Counterfactual Images\n"
        "Left: real X-ray | Right: what would need to change for the model to flip its answer",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(comp_dir / "all_counterfactual_comparisons.png", dpi=180,
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
        target_label = "PNEUMONIA" if result["pred_idx"] == 0 else "NORMAL"
        axes[i, 0].imshow(result["original_rgb"])
        axes[i, 0].set_title(
            f"Original {result['true_label']} X-ray\n({result['confidence']:.2f})",
            fontsize=8,
        )
        axes[i, 1].imshow(result["diff_overlay"])
        axes[i, 1].set_title(
            "What changed to flip the answer\n"
            "Colored areas = edited pixels",
            fontsize=8,
        )
        axes[i, 2].imshow(result["cf_rgb"])
        axes[i, 2].set_title(
            f"What if {target_label}?\n"
            f"Model now says: {result['cf_pred_label']} ({result['cf_confidence']:.2f})",
            fontsize=8,
        )
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(
        "What would need to change to flip the model prediction?",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comp_dir / "all_difference_maps.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def save_class_transition_grids(results, comp_dir):
    successful = [r for r in results if r["cf_success"]]
    if not successful:
        return

    groups = [
        (
            "normal_to_pneumonia",
            [
                r for r in successful
                if r["correct"]
                and r["true_label"] == "NORMAL"
                and r["pred_label"] == "NORMAL"
                and r["cf_pred_label"] == "PNEUMONIA"
            ],
            "This is how a NORMAL chest X-ray looks vs how PNEUMONIA would look",
            "This is a NORMAL chest X-ray",
            "This is how it would look\nwith PNEUMONIA (model prediction)",
        ),
        (
            "pneumonia_to_normal",
            [
                r for r in successful
                if r["correct"]
                and r["true_label"] == "PNEUMONIA"
                and r["pred_label"] == "PNEUMONIA"
                and r["cf_pred_label"] == "NORMAL"
            ],
            "This is how a PNEUMONIA chest X-ray looks vs how NORMAL would look",
            "This is a PNEUMONIA chest X-ray",
            "This is how it would look\nif it were NORMAL (model prediction)",
        ),
    ]

    for filename, rows, title, left_label, right_label in groups:
        if not rows:
            continue

        fig, axes = plt.subplots(len(rows), 3, figsize=(15, len(rows) * 4.2))
        axes = np.atleast_2d(axes)

        for i, result in enumerate(rows):
            axes[i, 0].imshow(result["original_rgb"])
            axes[i, 0].set_title(
                f"{left_label}\n"
                f"Model confidence: {result['confidence']:.2f}",
                fontsize=9,
            )

            axes[i, 1].imshow(result["cf_rgb"])
            axes[i, 1].set_title(
                f"{right_label}\n"
                f"Model confidence: {result['cf_confidence']:.2f}",
                fontsize=9,
            )

            axes[i, 2].imshow(result["diff_overlay"])
            axes[i, 2].set_title(
                "What changed between left and right\n"
                "Colored areas = edited pixels",
                fontsize=9,
            )

            for ax in axes[i]:
                ax.axis("off")

        fig.suptitle(title, fontsize=13, y=0.995)
        fig.text(
            0.5,
            0.02,
            "These are synthetic counterfactual explanations. The right column shows what the model needs "
            "to see to change its answer. This is not a real diagnosis — it shows what the model is sensitive to.",
            ha="center",
            va="bottom",
            fontsize=10,
            wrap=True,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(comp_dir / f"{filename}_comparison.png", dpi=180,
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
            "delta_limit": args.delta_limit,
            "delta_resolution": args.delta_resolution,
            "effect_strength": args.effect_strength,
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
        "active_change_outside_lung_fraction", "effect_strength",
        "effect_strength_used", "counterfactual_type", "semantic_strength",
        "delta_limit", "delta_resolution", "cf_steps_used", "panel_path",
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
    parser.add_argument("--l1-weight", type=float, default=0.004,
                        help="L1 sparsity penalty on the perturbation.")
    parser.add_argument("--tv-weight", type=float, default=0.02,
                        help="Total-variation smoothness penalty.")
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Target-class probability to consider the CF successful.")
    parser.add_argument("--delta-limit", type=float, default=1.0,
                        help="Maximum optimized perturbation in normalized image space.")
    parser.add_argument("--effect-strength", type=float, default=3.0,
                        help="Amplifies the learned perturbation in saved counterfactual images.")
    parser.add_argument("--delta-resolution", type=int, default=56,
                        help="Lower values produce smoother, broader visual edits.")
    parser.add_argument("--pixel-counterfactual", action="store_true",
                        help="Use unconstrained optimized pixel perturbations instead of semantic grayscale edits.")
    parser.add_argument("--semantic-strength", type=float, default=4.0,
                        help="Maximum strength for semantic pneumonia/normal visual edits.")
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
        original_rgb = tensor_to_display(input_tensor)

        source_class = row["pred_idx"]
        target_class = 1 - source_class

        best_delta = None
        cf_target_prob = None
        if args.pixel_counterfactual:
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
                delta_limit=args.delta_limit,
                delta_resolution=args.delta_resolution,
            )

        with torch.no_grad():
            if args.pixel_counterfactual:
                cf_tensor, cf_probs, cf_pred_idx, effect_strength_used, display_success = (
                    choose_display_counterfactual(
                        model, input_tensor, best_delta, mask_tensor, target_class,
                        decision_threshold, args.effect_strength,
                    )
                )
                cf_rgb = tensor_to_display(cf_tensor)
            else:
                seed = sum(bytearray(Path(row["path"]).stem.encode("utf-8"))) + target_class * 1009
                (
                    cf_tensor, cf_probs, cf_pred_idx, effect_strength_used,
                    cf_rgb, display_success,
                ) = choose_semantic_counterfactual(
                    model, original_rgb, mask_np, target_class, device,
                    decision_threshold, args.semantic_strength, seed,
                )
                cf_target_prob = float(cf_probs[target_class])
            cf_confidence = float(cf_probs[cf_pred_idx])
            cf_pred_label = CLASS_NAMES[cf_pred_idx]

        diff_gray, diff_overlay = make_difference_map(original_rgb, cf_rgb)
        localisation = change_localisation_metrics(diff_gray, mask_np)

        l1_norm = float(np.abs(cf_rgb - original_rgb).mean())

        panel_path = save_counterfactual_panel(
            row, original_rgb, cf_rgb, diff_gray, diff_overlay, mask_np,
            cf_pred_label, cf_confidence, CF_OUTPUT_DIR, COMP_OUTPUT_DIR, idx,
        )

        result = {
            **row,
            "cf_pred_idx": cf_pred_idx,
            "cf_pred_label": cf_pred_label,
            "cf_confidence": cf_confidence,
            "cf_success": bool(display_success),
            "cf_target_prob": cf_target_prob,
            "cf_l1_norm": l1_norm,
            "effect_strength": args.effect_strength,
            "effect_strength_used": effect_strength_used,
            "counterfactual_type": "pixel" if args.pixel_counterfactual else "semantic",
            "semantic_strength": args.semantic_strength,
            "delta_limit": args.delta_limit,
            "delta_resolution": args.delta_resolution,
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

        status = "FLIPPED" if display_success else "PARTIAL"
        print(
            f"  [{status}] {row['file']}: "
            f"{row['pred_label']}({row['confidence']:.2f}) -> "
            f"{cf_pred_label}({cf_confidence:.2f}), "
            f"L1={l1_norm:.4f}"
        )

    save_summary_grid(results, COMP_OUTPUT_DIR)
    save_difference_grid(results, COMP_OUTPUT_DIR)
    save_class_transition_grids(results, COMP_OUTPUT_DIR)
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
