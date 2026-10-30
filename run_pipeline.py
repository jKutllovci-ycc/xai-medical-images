"""
Master Pipeline Runner — Explainable AI for Medical Imaging

Runs the full pipeline from data download to explanation comparison.
Each step builds on the previous one:

  DATA
    STEP 1  Download Data              → Get chest X-ray images from Kaggle
    STEP 2  Explore & Preprocess       → Audit the dataset, compute statistics

  MODEL
    STEP 3  Train Classifier           → Fine-tune ResNet-18 for pneumonia detection
    STEP 4  Evaluate                   → Test set metrics, confusion matrix, ROC

  EXPLAIN
    STEP 5  Grad-CAM                   → Heatmaps: where does the model look?
    STEP 6  Counterfactual             → What would need to change to flip the diagnosis?
    STEP 7  Guided Counterfactual      → Grad-CAM + CF combined for focused changes

  EVALUATE & COMPARE
    STEP 8  Explanation Quality        → Faithfulness, SSIM, sparsity metrics
    STEP 9  Grad-CAM vs Counterfactual → Side-by-side comparison (key finding)

Usage:
    python run_pipeline.py                  # Run everything
    python run_pipeline.py --from 5         # Resume from step 5
    python run_pipeline.py --only 7 8       # Run only steps 7 and 8
    python run_pipeline.py --skip 1         # Skip step 1 
    python run_pipeline.py --dry-run        # Show what would run without executing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Pipeline stages and expected outputs
STEPS = [
    {
        "num": 1,
        "name": "Download Data",
        "script": "scripts/01_download_data.py",
        "description": "Downloads the Kaggle Chest X-Ray Pneumonia dataset.",
        "outputs": ["data/raw/chest_xray/"],
    },
    {
        "num": 2,
        "name": "Explore & Preprocess",
        "script": "scripts/02_preprocess.py",
        "description": "Audits the dataset: counts, statistics, manifests, and plots.",
        "outputs": ["experiments/results/dataset_manifest.csv", "experiments/results/data_summary.json", "experiments/results/*.png"],
    },
    {
        "num": 3,
        "name": "Train Classifier",
        "script": "scripts/03_train_classifier.py",
        "description": "Trains ResNet-18 with balanced sampling and early stopping.",
        "outputs": ["experiments/checkpoints/best_classifier.pt", "experiments/results/training_history.json", "experiments/results/training_curves.png"],
    },
    {
        "num": 4,
        "name": "Evaluate Model",
        "script": "scripts/04_evaluate.py",
        "description": "Runs inference on the test set and computes all metrics.",
        "outputs": ["experiments/results/test_metrics.json", "experiments/results/confusion_matrix.png", "experiments/results/roc_curve.png"],
    },
    {
        "num": 5,
        "name": "Grad-CAM Explanations",
        "script": "scripts/05_gradcam.py",
        "description": "Generates Grad-CAM++ heatmaps showing where the model looks.",
        "outputs": ["outputs/gradcam/"],
    },
    {
        "num": 6,
        "name": "Counterfactual Explanations",
        "script": "scripts/06_counterfactual.py",
        "args": ["--gan"],
        "description": "Creates normal and GAN counterfactuals in the same stage.",
        "outputs": ["outputs/counterfactuals/classic/", "outputs/counterfactuals/gan/", "outputs/comparisons/"],
    },
    {
        "num": 7,
        "name": "Grad-CAM-Guided Counterfactual",
        "script": "scripts/07_gradcam_guided_counterfactual.py",
        "description": "Uses Grad-CAM attention to focus CF perturbations for precision.",
        "outputs": ["outputs/guided_counterfactuals/"],
    },
    {
        "num": 8,
        "name": "Explanation Quality Metrics",
        "script": "scripts/08_explanation_quality.py",
        "description": "Measures faithfulness, SSIM, sparsity of all explanations.",
        "outputs": ["outputs/explanation_quality/"],
    },
    {
        "num": 9,
        "name": "Grad-CAM vs Counterfactual",
        "script": "scripts/09_compare_gradcam_counterfactual.py",
        "description": "Side-by-side comparison with overlap analysis.",
        "outputs": ["outputs/gradcam_vs_counterfactual/"],
    },
    {
        "num": 10,
        "name": "Clear Counterfactual Exports",
        "script": "scripts/10_export_clear_counterfactuals.py",
        "description": "Preserves source resolution and rechecks predictions on saved synthetic PNGs.",
        "outputs": ["outputs/clear_counterfactuals/"],
    },
]


def print_pipeline_overview():
    print()
    print("=" * 65)
    print("  EXPLAINABLE AI FOR MEDICAL IMAGING — FULL PIPELINE")
    print("=" * 65)
    print()
    print("  DATA       1. Download       →  2. Explore & Audit")
    print("                                       ↓")
    print("  MODEL      3. Train          →  4. Evaluate")
    print("                                       ↓")
    print("  EXPLAIN    5. Grad-CAM (baseline)")
    print("             6. Counterfactual (normal + GAN)")
    print("             7. Guided CF (Grad-CAM + CF combined)")
    print("                                       ↓")
    print("  EVALUATE   8. Quality Metrics →  9. Compare Methods")
    print()

 # Execute one pipeline stage
def run_step(step, dry_run=False):
    # This function runs one script and stops the pipeline if it fails.
    num = step["num"]
    name = step["name"]
    script = step["script"]
    command = [sys.executable, script, *step.get("args", [])]

    print(f"\n{'─' * 60}")
    print(f"  STEP {num:2d} │ {name}")
    print(f"         │ {step['description']}")
    print(f"         │ {script}")
    if step.get("outputs"):
        print("         │ outputs: " + ", ".join(step["outputs"]))
    print(f"{'─' * 60}")

    if dry_run:
        print("  [DRY RUN] Would execute: " + " ".join(command))
        return True

    if not Path(script).exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    start = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  STEP {num} COMPLETE ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  STEP {num} FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full XAI medical imaging pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from", type=int, dest="from_step", default=1,
                        help="Start from this step number (default: 1).")
    parser.add_argument("--to", type=int, default=10,
                        help="Stop after this step number (default: 10).")
    parser.add_argument("--only", type=int, nargs="+",
                        help="Run only these step numbers.")
    parser.add_argument("--skip", type=int, nargs="+", default=[],
                        help="Skip these step numbers.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing.")
    return parser.parse_args()


def main():
    # These options let me run all stages or restart from one stage.
    # Select and run stages in order
    args = parse_args()
    print_pipeline_overview()

    if args.only:
        steps_to_run = [s for s in STEPS if s["num"] in args.only]
    else:
        steps_to_run = [
            s for s in STEPS
            if args.from_step <= s["num"] <= args.to
            and s["num"] not in args.skip
        ]

    if not steps_to_run:
        print("No steps to run.")
        return

    print(f"\nWill run {len(steps_to_run)} step(s): "
          + ", ".join(f'{s["num"]}' for s in steps_to_run))

    if args.dry_run:
        print("[DRY RUN MODE]")

    total_start = time.time()
    passed = 0
    failed = 0

    for step in steps_to_run:
        success = run_step(step, dry_run=args.dry_run)
        if success:
            passed += 1
        else:
            failed += 1
            if not args.dry_run:
                print(f"\nPipeline stopped at step {step['num']}. "
                      f"Fix the issue and rerun with:")
                print(f"  python run_pipeline.py --from {step['num']}")
                break

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"  Passed: {passed} | Failed: {failed} | Time: {total_elapsed:.1f}s")
    if failed == 0 and not args.dry_run:
        print("\n  All steps complete! Review the final comparison outputs:")
        print("    outputs/gradcam_vs_counterfactual/master_comparison_grid.png")
        print("    notebooks/results.ipynb")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
