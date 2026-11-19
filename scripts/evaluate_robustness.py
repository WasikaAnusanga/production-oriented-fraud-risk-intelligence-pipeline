"""
Execution Script for Stage 11: Robustness and Stress Testing (Section J).

Runs:
1. Baseline evaluation on Policy-Validation cohort.
2. Scenarios 1 & 2: Monetary inflation (+25%, +50%).
3. Scenario 3: Consistent velocity surge.
4. Scenario 4: Digital channel shift.
5. Scenario 5: Additional numerical missingness.
6. Scenario 6: Missing authentication level spike.
7. Scenario 7: Extreme telemetry outliers.
8. Scenario 8: Unseen categorical drift.
9. Scenario 9: 5-fold cross-validation performance and cost stability.
10. Exports reports/tables/stress_test_results.csv and reports/tables/fold_stability_results.csv.
11. Renders publication figures: 21_stress_testing_workload_and_cost.png,
    22_stress_testing_probability_drift.png, and 23_cross_validation_fold_stability.png.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config.settings import RANDOM_STATE
from src.data_splitting import get_partition_data
from src.cost_optimization import load_business_cost_config
from src.robustness import (
    run_comprehensive_stress_testing,
    evaluate_development_fold_stability,
    plot_stress_test_visualizations,
)


def run_stage_11():
    print("=" * 70)
    print("STAGE 11: ROBUSTNESS & STRESS TESTING (SECTION J)")
    print("=" * 70)

    # 1. Load Policy-Validation cohort (n=600)
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    print(f"Loaded Policy-Validation cohort: {len(X_val)} transactions, {y_val.sum()} frauds ({y_val.mean()*100:.2f}%)")

    # 2. Load Development cohort (n=1,800) for Fold Stability (Scenario 9)
    X_dev, y_dev, ids_dev = get_partition_data(partition="development")
    print(f"Loaded Development cohort: {len(X_dev)} transactions, {y_dev.sum()} frauds ({y_dev.mean()*100:.2f}%)")

    # 3. Load Frozen Calibrated Production Model
    model_path = "04_calibrated_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Calibrated production model not found at {model_path}")
    calibrated_model = joblib.load(model_path)
    print(f"Loaded frozen production model from {model_path}")

    # Load business cost configuration (Scenario B Primary Baseline)
    cost_cfg = load_business_cost_config("config/business_costs.yaml")
    primary_scenario_cfg = cost_cfg["scenarios"]["scenario_b_balanced"]

    # -------------------------------------------------------------
    # 4. Execute Scenarios 1 to 8: Stress Testing Grid
    # -------------------------------------------------------------
    print("\nExecuting Comprehensive Stress Testing Scenarios (1 to 8)...")
    stress_df = run_comprehensive_stress_testing(
        calibrated_model=calibrated_model,
        X_val=X_val,
        y_val=y_val,
        scenario_cfg=primary_scenario_cfg,
    )

    print("\nStress Testing Results Summary:")
    disp_cols = [
        "scenario",
        "mean_probability",
        "pct_review",
        "pct_block",
        "fraud_capture_pct",
        "cost_per_1k",
        "roc_auc",
        "pr_auc",
        "wasserstein_drift",
    ]
    print(stress_df[disp_cols].to_string(index=False))

    # -------------------------------------------------------------
    # 5. Scenario 9: Development Fold Stability Analysis
    # -------------------------------------------------------------
    print("\nExecuting Scenario 9: 5-Fold Cross-Validation Calibrator Stability...")
    fold_df = evaluate_development_fold_stability(
        calibrated_model=calibrated_model,
        X_dev=X_dev,
        y_dev=y_dev,
        scenario_cfg=primary_scenario_cfg,
    )
    print("\nFold Stability Results:")
    print(fold_df.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Export Tables
    # -------------------------------------------------------------
    os.makedirs("reports/tables", exist_ok=True)
    stress_df.to_csv("reports/tables/stress_test_results.csv", index=False)
    fold_df.to_csv("reports/tables/fold_stability_results.csv", index=False)
    print("\nSaved stress test tables to reports/tables/.")

    # -------------------------------------------------------------
    # 7. Render Visualizations (300 DPI)
    # -------------------------------------------------------------
    print("\nRendering Stress Testing Visualizations...")
    figs = plot_stress_test_visualizations(
        stress_results_df=stress_df,
        fold_stability_df=fold_df,
        output_dir="reports/figures",
    )
    for name, p in figs.items():
        print(f"  Generated [{name}]: {p}")

    print("\nStage 11 Robustness & Stress Testing completed successfully!")


if __name__ == "__main__":
    run_stage_11()
