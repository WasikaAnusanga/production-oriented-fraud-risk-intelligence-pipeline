"""
Evaluation and Comparison Script for Stage 7: Probability Calibration.

Orchestrates:
1. Loading the frozen Development partition (1,800 records).
2. Loading the primary candidate model pipeline (RF_Tuned) and secondary benchmarks (LR_Tuned, HGB_Tuned).
3. Running out-of-fold cross-validation for Uncalibrated, Sigmoid, and Isotonic calibration.
4. Exporting reports/tables/calibration_comparison.csv.
5. Generating all 5 required calibration visualizations at 300 DPI.
6. Fitting and serializing the final calibrated production candidate wrapping the end-to-end pipeline.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config.settings import RANDOM_STATE, RAW_DATA_PATH
from src.data_splitting import get_partition_data
from src.calibration import (
    cross_validate_calibration,
    fit_final_calibrated_pipeline,
    plot_reliability_diagrams,
    evaluate_calibration_metrics,
)


def run_calibration_evaluation():
    print("=" * 70)
    print("STAGE 7: PROBABILITY CALIBRATION & RELIABILITY ENGINEERING")
    print("=" * 70)

    # 1. Load Development partition ONLY
    X_dev, y_dev, ids_dev = get_partition_data(partition="development")
    print(f"Loaded Development partition: {X_dev.shape[0]} rows, {y_dev.sum()} frauds ({y_dev.mean()*100:.2f}%)")

    # 2. Load candidate pipeline (RF_Tuned)
    candidate_path = "reports/intermediate/candidate_rf_pipeline.joblib"
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate pipeline not found at {candidate_path}")
    
    rf_pipeline = joblib.load(candidate_path)
    print(f"Loaded candidate pipeline from {candidate_path}")

    # 3. Perform 5-fold cross-validation of calibration methods
    print("\nRunning 5-fold Out-of-Fold Calibration Evaluation (Uncalibrated vs Sigmoid vs Isotonic)...")
    summary_df, oof_preds, fold_df = cross_validate_calibration(
        base_pipeline=rf_pipeline,
        X=X_dev,
        y=y_dev,
        n_splits=5,
        random_state=RANDOM_STATE,
    )

    print("\nCalibration Comparison Summary (5-Fold Out-of-Fold Development CV):")
    print(summary_df.to_string(index=False))

    # Also evaluate over the entire concatenated out-of-fold predictions
    oof_metrics_list = []
    for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
        metrics = evaluate_calibration_metrics(y_dev.to_numpy(), oof_preds[method])
        oof_metrics_list.append({"Method": method, **metrics})
    oof_metrics_df = pd.DataFrame(oof_metrics_list)
    print("\nConcatenated Out-of-Fold Performance:")
    print(oof_metrics_df.to_string(index=False))

    # 4. Save results to reports/tables/calibration_comparison.csv
    os.makedirs("reports/tables", exist_ok=True)
    summary_df.to_csv("reports/tables/calibration_comparison.csv", index=False)
    fold_df.to_csv("reports/tables/calibration_folds_detail.csv", index=False)
    oof_metrics_df.to_csv("reports/tables/calibration_oof_metrics.csv", index=False)
    print("\nSaved calibration comparison tables to reports/tables/")

    # 5. Generate Visualizations (300 DPI)
    print("\nGenerating publication-quality calibration plots...")
    figs = plot_reliability_diagrams(
        y_true=y_dev.to_numpy(),
        oof_predictions=oof_preds,
        output_dir="reports/figures",
    )
    for name, path in figs.items():
        print(f"  Generated [{name}]: {path}")

    # 6. Fit Final Calibrated Production Model on entire Development Partition
    # We choose Sigmoid (Platt) scaling because:
    # - Isotonic regression is unconstrained and highly susceptible to overfitting/step-function degradation on small sample sizes (only 218 fraud cases in dev).
    # - Sigmoid provides smooth, monotonic, well-regularized probabilities that protect against zero probability overconfidence.
    print("\nFitting final calibrated pipeline (Sigmoid/Platt scaling, 5-fold CV) on entire Development partition...")
    calibrated_pipeline = fit_final_calibrated_pipeline(
        base_pipeline=rf_pipeline,
        X_dev=X_dev,
        y_dev=y_dev,
        method="sigmoid",
        cv=5,
    )

    # Verify input acceptance on raw DataFrame
    test_sample = X_dev.head(5)
    sample_probs = calibrated_pipeline.predict_proba(test_sample)[:, 1]
    print(f"Verification: raw input inference successful. First 5 calibrated fraud probabilities:\n  {sample_probs.round(4)}")

    # 7. Save intermediate calibrated candidate
    os.makedirs("reports/intermediate", exist_ok=True)
    intermediate_path = "reports/intermediate/calibrated_candidate_pipeline.joblib"
    joblib.dump(calibrated_pipeline, intermediate_path)
    print(f"Saved calibrated candidate pipeline to {intermediate_path}")

    # Root intermediate requirement: 04_calibrated_model.joblib
    root_calibrated_path = "04_calibrated_model.joblib"
    joblib.dump(calibrated_pipeline, root_calibrated_path)
    print(f"Saved production calibrated pipeline to {root_calibrated_path}")

    print("\nStage 7 probability calibration evaluation completed successfully!")


if __name__ == "__main__":
    run_calibration_evaluation()
