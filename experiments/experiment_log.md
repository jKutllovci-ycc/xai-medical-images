# Experiment Log

## Format
Each experiment entry records: date, settings, results, and observations.
This ensures all findings are fully reproducible and auditable.

---

### Experiment 001 — Baseline Classifier v1
- **Date:** 
- **Model:** ResNet-18, pretrained ImageNet
- **Settings:** lr=0.0001, batch=32, epochs=25, dropout=0.3
- **Augmentation:** HFlip, Rotation(10), ColorJitter(0.1, 0.1)
- **Results:**
  - Train Acc: 
  - Val Acc: 
  - Test Acc: 
  - F1: 
  - AUC-ROC: 
- **Observations:** 

---

### Experiment 002 — Classifier v2 (Improved)
- **Date:** 2026-07-19
- **Changes from v1:**
  - Replaced naive shuffled training with a weighted random sampler to reduce pneumonia-class bias from the 74.3% pneumonia training split.
  - Removed horizontal flipping because chest X-rays contain laterality markers and asymmetric anatomy.
  - Switched to safer medical-image augmentation: resize/crop, mild affine transforms, and mild brightness/contrast jitter.
  - Added label smoothing, AdamW, differential learning rates for pretrained backbone vs classifier head, gradient clipping, and balanced-accuracy checkpointing.
  - Expanded preprocessing audit with dataset manifest, duplicate detection, unreadable-image checks, and recommended class weights.
  - Added Grad-CAM before/after comparison panels showing the original vanilla Grad-CAM output beside the revised masked Grad-CAM++ output for the same case.
- **Results:**
  - Pending retraining. Run `python3 scripts/03_train_classifier.py`, then `python3 scripts/04_evaluate.py` and `python3 scripts/05_gradcam.py`.
- **Observations:**
  - Baseline test metrics showed high pneumonia recall but weak NORMAL recall, suggesting false-positive bias.
  - Grad-CAM review showed several hot regions outside clinically convincing parenchymal findings, so classifier training quality needs improvement in addition to heatmap postprocessing.

---

(Continue logging each experiment iteration)
