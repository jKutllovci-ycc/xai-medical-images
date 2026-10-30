"""Export native-resolution research counterfactuals and re-score saved PNGs.

Transfer only the low-resolution intensity change to the source image. Source
detail is retained, but this does not recover anatomy hidden by disease or
establish that the synthetic image is clinically plausible.
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location("counterfactual", Path(__file__).with_name("06_counterfactual.py"))
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)
import matplotlib.pyplot as plt

OUTPUT = Path("outputs/clear_counterfactuals")


def transfer_edit(native_rgb, model_original, model_cf):
    # Resize only the learned change and add it to the original full-size image.
    # Transfer the model-space edit to source resolution
    delta = (model_cf - model_original).mean(axis=2)
    native_delta = cv2.resize(delta, (native_rgb.shape[1], native_rgb.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
    return np.clip(native_rgb + native_delta[..., None], 0, 1)


def main():
    # Save clear PNG files, reopen them, and verify that every saved image still reaches its target.
    # Export and verify saved counterfactual images
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = cf.get_device()
    model = cf.build_model(device)
    mode = cf.load_preprocess_mode_from_checkpoint(cf.CHECKPOINT_PATH)
    cf.CURRENT_PREPROCESS_MODE = mode
    transform = cf.build_eval_transform()
    threshold = cf.load_decision_threshold()
    metadata = json.loads((cf.CF_OUTPUT_DIR / "counterfactual_metadata.json").read_text())
    exported = []
    preview_rows = {0: [], 1: []}
    for index, row in enumerate(metadata["selected"]):
        target = 1 - row["true_idx"]
        stem, target_label = cf.make_descriptive_stem(row, index)
        with Image.open(row["path"]) as source:
            source = source.convert("RGB")
            native = np.asarray(cf.apply_image_preprocessing(source, mode), dtype=np.float32) / 255
            original = cf.tensor_to_display(transform(source).unsqueeze(0))
        with Image.open(cf.CF_OUTPUT_DIR / f"{stem}_edited_to_look_{target_label}.png") as im:
            edited = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        high = transfer_edit(native, original, edited)
        original_path = OUTPUT / f"{stem}_original.png"
        image_path = OUTPUT / f"{stem}_synthetic_{target_label}.png"
        Image.fromarray(np.uint8(np.rint(native * 255))).save(original_path)
        Image.fromarray(np.uint8(np.rint(high * 255))).save(image_path)
        # Exports are already in checkpoint preprocessing space; normalize once.
        with Image.open(image_path) as saved:
            resized = cf.transforms.Resize((cf.IMG_SIZE, cf.IMG_SIZE))(saved.convert("RGB"))
            tensor = cf.display_to_tensor(np.asarray(resized, dtype=np.float32) / 255, device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        pred = cf.predict_label_from_probs(probs, threshold)
        panel = OUTPUT / f"{stem}_comparison.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
        for ax, im, title in zip(axes, [native, high], [
            f"Original: {row['true_label']}",
            f"Synthetic target: {cf.CLASS_NAMES[target]}\nModel: {cf.CLASS_NAMES[pred]} | P(pneumonia)={probs[1]:.3f}",
        ]):
            ax.imshow(im, vmin=0, vmax=1)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig.suptitle("Research counterfactual | Clinical appearance not validated", fontsize=11)
        fig.savefig(panel, dpi=200)
        plt.close(fig)
        entry = {
            "file": row["file"], "source_label": row["true_label"],
            "target_label": cf.CLASS_NAMES[target], "saved_prediction": cf.CLASS_NAMES[pred],
            "saved_target_reached": pred == target, "saved_pneumonia_probability": float(probs[1]),
            "width": high.shape[1], "height": high.shape[0],
            "preprocess_mode": mode, "original_path": str(original_path),
            "image_path": str(image_path), "panel_path": str(panel),
        }
        exported.append(entry)
        if len(preview_rows[target]) < 3:
            preview_rows[target].append((native, high, entry))
        print(f"{row['file']}: {row['true_label']} -> {cf.CLASS_NAMES[pred]}, {high.shape[1]}x{high.shape[0]}", flush=True)
    for target, rows in preview_rows.items():
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 2, figsize=(10, len(rows) * 4.5), squeeze=False, layout="constrained")
        for axes_row, (original, edited, entry) in zip(axes, rows):
            for ax, im, title in zip(axes_row, [original, edited], [
                f"Original {entry['source_label']}",
                f"Synthetic target {entry['target_label']}\nModel: {entry['saved_prediction']}",
            ]):
                ax.imshow(im)
                ax.set_title(title, fontsize=11)
                ax.axis("off")
        fig.suptitle("Synthetic research images | Clinical appearance not validated", fontsize=11)
        direction = "normal_to_pneumonia" if target else "pneumonia_to_normal"
        fig.savefig(OUTPUT / f"{direction}.png", dpi=160)
        plt.close(fig)
    (OUTPUT / "export_metadata.json").write_text(json.dumps({
        "method": "Native source detail plus interpolated grayscale model-space residual",
        "clinical_validation": False, "decision_threshold": threshold, "selected": exported,
    }, indent=2))


if __name__ == "__main__":
    main()
