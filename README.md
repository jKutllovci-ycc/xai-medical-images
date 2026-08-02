# Explainable AI for Medical Imaging
## Counterfactual Explanations for Pneumonia Detection in Chest X-Ray

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Kaggle dataset (needs API key from kaggle.com -> Account -> Create New API Token)
python scripts/01_download_data.py

# 3. Explore the data — saves stats & plots
python scripts/02_preprocess.py

# 4. Train the pneumonia classifier
python scripts/03_train_classifier.py

# 5. Evaluate on test set
python scripts/04_evaluate.py

# 6. Generate Grad-CAM explanations and diagnostics
python scripts/05_gradcam.py

# Optional: explain predicted classes, or compare CAM layers
python scripts/05_gradcam.py --target-class predicted
python scripts/05_gradcam.py --target-layer layer3
python scripts/05_gradcam.py --no-comparison

# 7. View all results in the notebook
jupyter notebook notebooks/results.ipynb
```

### Project Structure

```
xai-medical-images/
├── scripts/                        # Run these in order
│   ├── 01_download_data.py         # Download Kaggle dataset
│   ├── 02_preprocess.py            # Data exploration & stats
│   ├── 03_train_classifier.py      # Train ResNet-18 classifier
│   ├── 04_evaluate.py              # Test set evaluation
│   └── 05_gradcam.py               # Grad-CAM heatmap explanations
├── notebooks/
│   └── results.ipynb               # View all findings & plots
├── data/raw/chest_xray/            # Dataset (train/val/test)
├── experiments/
│   ├── checkpoints/                # Saved model weights
│   ├── results/                    # Metrics, plots, JSON outputs
│   └── experiment_log.md           # Manual experiment notes
└── outputs/
    ├── gradcam/                    # Grad-CAM heatmap images
    ├── counterfactuals/            # (future) Generated counterfactuals
    └── comparisons/                # (future) Side-by-side comparisons
```

### Dataset

Kaggle Chest X-Ray Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
