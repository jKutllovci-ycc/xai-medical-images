# Explainable AI for Medical Imaging
## Counterfactual Explanations for Pneumonia Detection in Chest X-Rays

### Pipeline Overview

```
  DATA       1. Download       →  2. Explore & Audit
                                        ↓
  MODEL      3. Train          →  4. Evaluate
                                        ↓
  EXPLAIN    5. Grad-CAM (baseline — where does the model look?)
             6. Counterfactual (main — what would flip the diagnosis?)
             7. Guided CF (Grad-CAM + CF combined for precision)
                                        ↓
  EVALUATE   8. Quality Metrics →  9. Compare Grad-CAM vs CF
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
python run_pipeline.py --only 7 8 9 

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
python scripts/06_counterfactual.py

# Optional: more steps or different regularisation
python scripts/06_counterfactual.py --steps 800 --max-per-class 8

# 7. Grad-CAM-guided counterfactuals — uses Grad-CAM attention to focus CF
#    perturbations only on regions the model actually uses (more precise)
python scripts/07_gradcam_guided_counterfactual.py

# Optional: adjust how tightly Grad-CAM focuses the mask
python scripts/07_gradcam_guided_counterfactual.py --cam-threshold 0.2 --blend-weight 0.8

# 8. Explanation quality metrics — faithfulness, SSIM, sparsity
python scripts/08_explanation_quality.py

# 9. Grad-CAM vs Counterfactual comparison — side-by-side with overlap analysis
python scripts/09_compare_gradcam_counterfactual.py
```

### What Each Step Shows

| Step | What it answers | Output |
|------|----------------|--------|
| **05 Grad-CAM** | Where does the model look? | Heatmap overlays showing attention regions |
| **06 Counterfactual** | What would flip the diagnosis? | Modified images + difference maps |
| **07 Guided CF** | Can we make CFs more precise? | Focused changes on attention regions only |
| **08 Quality** | Are the explanations faithful? | Insertion/deletion curves, SSIM, sparsity |
| **09 Compare** | Do Grad-CAM and CF agree? | Side-by-side panels, overlap analysis |


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
    └── gradcam_vs_counterfactual/          # Side-by-side comparison panels
```

### Dataset

Kaggle Chest X-Ray Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
