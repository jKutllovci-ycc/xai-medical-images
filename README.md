# Explainable AI for Medical Imaging
## Counterfactual Explanations for Pneumonia Detection in Chest X-Rays

### Pipeline Overview

```
  DATA       1. Download       →  2. Explore & Audit
                                        ↓
  MODEL      3. Train          →  4. Evaluate
                                        ↓
  EXPLAIN    5. Grad-CAM (baseline — where does the model look?)
             6. Counterfactual (normal + GAN — what would flip the diagnosis?)
             7. Guided CF (Grad-CAM + CF combined for precision)
                                        ↓
  EVALUATE   8. Quality Metrics →  9. Compare Grad-CAM vs CF
                                        ↓
  EXPORT     10. Clear native-resolution counterfactual images
```

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the entire pipeline
python run_pipeline.py

# Or resume from a specific step (e.g. skip data download and training)
python run_pipeline.py --from 5

# Or run only specific steps
python run_pipeline.py --only 7 8 9 10

# Dry run to see what would execute
python run_pipeline.py --dry-run
```

### Step-by-Step

```bash
# 1. Download Kaggle dataset (needs API key: kaggle.com -> Account -> Create New API Token)
python scripts/01_download_data.py

# 2. Explore the data — saves stats, manifests & plots
python scripts/02_preprocess.py

# 3. Train the pneumonia classifier (ResNet-18, ~25 epochs with early stopping)
python scripts/03_train_classifier.py

# 4. Evaluate on test set — confusion matrix, ROC curve, all metrics
python scripts/04_evaluate.py

# 5. Grad-CAM explanations — heatmaps showing where the model looks (baseline)
python scripts/05_gradcam.py

# Optional Grad-CAM variants:
python scripts/05_gradcam.py --target-class predicted
python scripts/05_gradcam.py --target-layer layer3

# 6. Counterfactual explanations — what would need to change to flip the diagnosis?
python scripts/06_counterfactual.py --gan

# Optional: more steps or different regularisation
python scripts/06_counterfactual.py --steps 800 --max-per-class 8

# 7. Grad-CAM-guided counterfactuals — uses Grad-CAM attention to focus CF
#    perturbations only on regions the model actually uses (more precise)
python scripts/07_gradcam_guided_counterfactual.py

# Optional: adjust how tightly Grad-CAM focuses the mask
python scripts/07_gradcam_guided_counterfactual.py --cam-threshold 0.2 --blend-weight 0.8

# 8. Explanation quality metrics — audits classic, guided, and GAN outputs
python scripts/08_explanation_quality.py

# 9. Grad-CAM vs Counterfactual comparison — side-by-side with overlap analysis
python scripts/09_compare_gradcam_counterfactual.py

# 10. Native-resolution originals, synthetic edits, and uncluttered comparisons
python scripts/10_export_clear_counterfactuals.py
```

### GAN Counterfactual Method Inside Stage 6

Stage 6 keeps the normal counterfactual method and also trains an unpaired
CycleGAN-style model on the training split only when `--gan` is used.
`G_N2P` learns NORMAL to PNEUMONIA and `G_P2N` learns PNEUMONIA to NORMAL.
Two PatchGAN discriminators compete with these generators. Cycle-consistency and
identity losses limit structural changes, while the frozen ResNet-18 supplies a
target-class loss. The same stage applies the trained generators to the selected
cases used by the other explanation methods and saves probability, success, SSIM,
and image-change measurements in `outputs/counterfactuals/gan/`. The existing
method is saved separately in `outputs/counterfactuals/classic/`.

This is an experimental explanation method, not a clinical disease simulator.
The master pipeline remains ten stages. GAN training is part of stage 6 and is
more expensive than the normal counterfactual method. A short engineering check
can be run with:

```bash
python scripts/06_counterfactual.py --gan --gan-retrain --gan-epochs 1 \
  --gan-max-steps 1 --gan-image-size 64 --gan-channels 8 --gan-blocks 1
```

### Clear Counterfactual Images

Stage 10 is a post-processing and export stage, not another explanation method.
It reads the examples and edited images produced by stage 6, then transfers each
224 x 224 grayscale intensity change onto the corresponding source-resolution
X-ray. This produces clearer presentation images while retaining the original
image dimensions, detail, and aspect ratio. It does not generate new anatomical
detail or retrain either model.

After saving each native-resolution synthetic PNG, stage 10 resizes it back to
the classifier input size and scores it again. This verifies whether the exported
file still reaches the intended target class after resizing and PNG encoding.
The outputs include:

- native-resolution copies of the original X-rays;
- native-resolution synthetic counterfactual X-rays;
- uncluttered original-versus-synthetic comparison panels;
- overview grids for NORMAL → PNEUMONIA and PNEUMONIA → NORMAL; and
- `outputs/clear_counterfactuals/export_metadata.json`, containing dimensions,
  predictions, probabilities, and whether each saved image reached its target.

Stage 10 requires the classifier checkpoint and the classic counterfactual
images plus metadata from stage 6. If those already exist, it can be run by
itself; it does not require rerunning stages 7–9.

Open `outputs/clear_counterfactuals/normal_to_pneumonia.png` and
`outputs/clear_counterfactuals/pneumonia_to_normal.png` for previews, or the
individual `*_comparison.png` files for larger pairs. These are synthetic research
explanations: a target prediction does not establish a medically realistic change
in disease. The lung masks are image-processing estimates, not validated segmentation.

To regenerate the shared examples and all explanation outputs with the existing
checkpoint:

```bash
python scripts/06_counterfactual.py --reselect
python run_pipeline.py --only 5 7 8 9 10
```

### What Each Step Shows

| Step | What it answers | Output |
|------|----------------|--------|
| **05 Grad-CAM** | Where does the model look? | Heatmap overlays showing attention regions |
| **06 Counterfactual** | What would flip the diagnosis? | Modified images + difference maps |
| **07 Guided CF** | Can we make CFs more precise? | Focused changes on attention regions only |
| **08 Quality** | Are the explanations faithful? | Insertion/deletion curves, SSIM, sparsity |
| **09 Compare** | Do Grad-CAM and CF agree? | Side-by-side panels, overlap analysis |
| **10 Clear exports** | Do the counterfactual edits remain effective when presented at the source image size? | Native-resolution originals and synthetic images, comparison panels, re-scored export metadata |


### Project Structure

```
xai-medical-images/
├── run_pipeline.py                         # Master pipeline runner
├── scripts/                                # Run these in order (or use run_pipeline.py)
│   ├── 01_download_data.py                 # Download Kaggle dataset
│   ├── 02_preprocess.py                    # Data exploration & stats
│   ├── 03_train_classifier.py              # Train ResNet-18 classifier
│   ├── 04_evaluate.py                      # Test set evaluation
│   ├── 05_gradcam.py                       # Grad-CAM heatmap explanations
│   ├── 06_counterfactual.py                # Counterfactual explanations
│   ├── 07_gradcam_guided_counterfactual.py # Grad-CAM-guided counterfactuals
│   ├── 08_explanation_quality.py           # Explanation quality metrics
│   ├── 09_compare_gradcam_counterfactual.py # Grad-CAM vs generator
│   └── 10_export_clear_counterfactuals.py   # Native-resolution CF exports and verification
├── data/raw/chest_xray/                    # Dataset (train/val/test)
├── experiments/
│   ├── checkpoints/                        # Saved model weights
│   ├── results/                            # Metrics, plots, JSON outputs
│   └── experiment_log.md                   # Manual experiment notes
└── outputs/
    ├── gradcam/                            # Grad-CAM heatmap images
    ├── counterfactuals/                    # Counterfactual images & diff maps
    ├── comparisons/                        # Counterfactual comparison panels
    ├── guided_counterfactuals/             # Grad-CAM-guided CF outputs
    ├── explanation_quality/                # Faithfulness & quality charts
    ├── gradcam_vs_counterfactual/          # Side-by-side comparison panels
    └── clear_counterfactuals/              # Native-resolution exports and metadata
```

### Dataset

Kaggle Chest X-Ray Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
