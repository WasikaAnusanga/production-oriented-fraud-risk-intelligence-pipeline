"""
Execution Script for Stage 9: Explainable AI & Regulatory Reason Codes (Section H).

Uses Development & Policy-Validation data only.
- Computes global feature importance via TreeSHAP and Gini impurity.
- Evaluates feature importance stability across the 5 cross-validation calibrator folds.
- Renders ranked importance bar chart, SHAP summary beeswarm plot, and 3 dependence plots.
- Generates 5 local SHAP waterfall plots and adverse action explanation records across risk regimes.
- Exports reports/tables/global_feature_importance.csv, reports/tables/local_explanations.csv,
  and reports/tables/feature_importance_stability.csv.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config.settings import RANDOM_STATE
from src.data_splitting import get_partition_data
from src.explainability import (
    compute_global_feature_importance,
    analyze_feature_importance_stability,
    plot_global_importance_and_summary,
    plot_shap_dependences,
    generate_local_explanations,
)


def run_stage_9():
    print("=" * 70)
    print("STAGE 9: EXPLAINABLE ARTIFICIAL INTELLIGENCE (SECTION H)")
    print("=" * 70)

    # 1. Load Development Partition (n=1,800)
    X_dev, y_dev, ids_dev = get_partition_data(partition="development")
    print(f"Loaded Development partition: {X_dev.shape[0]} rows, {y_dev.sum()} frauds ({y_dev.mean()*100:.2f}%)")

    # 2. Load Policy-Validation Partition (n=600)
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    print(f"Loaded Policy-Validation partition: {X_val.shape[0]} rows, {y_val.sum()} frauds ({y_val.mean()*100:.2f}%)")

    # 3. Load Frozen Production Models
    model_path = "04_calibrated_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Calibrated production model not found at {model_path}")
    calibrated_model = joblib.load(model_path)
    print(f"Loaded production calibrated model from {model_path}")

    # Extract primary uncalibrated base pipeline from fold 0 estimator
    base_pipeline = calibrated_model.calibrated_classifiers_[0].estimator
    print(f"Extracted base estimator pipeline: {type(base_pipeline)}")

    # -------------------------------------------------------------
    # 4. Global Feature Importance (TreeSHAP & Gini)
    # -------------------------------------------------------------
    print("\nComputing Global Feature Importance via TreeSHAP on Development Cohort...")
    # Use a representative sample of 600 development rows for swift TreeSHAP computation
    dev_sample_X = X_dev.sample(n=600, random_state=RANDOM_STATE)
    dev_sample_y = y_dev.loc[dev_sample_X.index]

    importance_df, shap_matrix, shap_exp, X_trans = compute_global_feature_importance(
        pipeline=base_pipeline,
        X_sample=dev_sample_X,
        y_sample=dev_sample_y,
        n_permutations=3,
        random_state=RANDOM_STATE,
    )

    print("\nTop 15 Global Predictors by Mean |SHAP|:")
    print(importance_df.head(15)[["shap_rank", "feature", "mean_abs_shap", "gini_importance", "is_engineered"]].to_string(index=False))

    # -------------------------------------------------------------
    # 5. Feature Importance Stability Across Folds
    # -------------------------------------------------------------
    print("\nAnalyzing Feature Importance Stability Across 5 Cross-Validation Calibrator Folds...")
    stability_df = analyze_feature_importance_stability(
        calibrated_model=calibrated_model,
        X_dev=X_dev,
        y_dev=y_dev,
        top_k=15,
    )
    print("\nTop Predictors Stability Across Folds (Mean, Std, CV%):")
    print(stability_df[["rank", "feature", "mean_importance", "std_importance", "cv_pct_stability"]].to_string(index=False))

    # -------------------------------------------------------------
    # 6. Global Visualizations (300 DPI)
    # -------------------------------------------------------------
    print("\nRendering Global Explainability Visualizations...")
    os.makedirs("reports/figures", exist_ok=True)
    global_figs = plot_global_importance_and_summary(
        importance_df=importance_df,
        shap_explanation=shap_exp,
        output_dir="reports/figures",
        top_n=15,
    )
    for name, p in global_figs.items():
        print(f"  Generated [{name}]: {p}")

    dep_figs = plot_shap_dependences(
        shap_explanation=shap_exp,
        output_dir="reports/figures",
    )
    for name, p in dep_figs.items():
        print(f"  Generated [{name}]: {p}")

    # -------------------------------------------------------------
    # 7. Local Explainability & Case Studies
    # -------------------------------------------------------------
    print("\nGenerating 5 Local Explainability Case Studies & Waterfall Plots...")
    local_df, local_figs = generate_local_explanations(
        pipeline=base_pipeline,
        calibrated_pipeline=calibrated_model,
        X_cohort=X_val,
        y_cohort=y_val,
        ids_cohort=ids_val,
        output_dir="reports/figures",
        review_threshold=0.10,
        block_threshold=0.36,
    )
    for name, p in local_figs.items():
        print(f"  Generated [{name}]: {p}")

    print("\nLocal Explanation Summary:")
    print(local_df[["case_type", "transaction_id", "actual_fraud", "calibrated_prob", "operational_action"]].to_string(index=False))

    # -------------------------------------------------------------
    # 8. Export Tables
    # -------------------------------------------------------------
    os.makedirs("reports/tables", exist_ok=True)
    importance_df.to_csv("reports/tables/global_feature_importance.csv", index=False)
    stability_df.to_csv("reports/tables/feature_importance_stability.csv", index=False)
    local_df.to_csv("reports/tables/local_explanations.csv", index=False)
    print("\nSaved all explainability tables to reports/tables/.")

    print("\nStage 9 Explainable AI completed successfully!")


if __name__ == "__main__":
    run_stage_9()
