"""
Execution Script for Stage 8: Cost-Sensitive Learning & Threshold Optimization.

Strict Data Governance:
- Uses ONLY the 600-row Policy-Validation partition.
- Uses the frozen calibrated pipeline from Stage 7 (04_calibrated_model.joblib).
- NEVER exposes or evaluates the independent final test set.
- Exports 05_threshold_cost_analysis.csv, reports/tables/business_cost_comparison.csv,
  reports/tables/cost_sensitivity.csv, and all 4 publication figures.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config.settings import RANDOM_STATE
from src.data_splitting import get_partition_data
from src.cost_optimization import (
    load_business_cost_config,
    evaluate_policy_costs,
    grid_search_thresholds,
    run_cost_sensitivity_analysis,
    plot_threshold_cost_surfaces,
    plot_sensitivity_heatmaps,
)


def run_stage_8():
    print("=" * 70)
    print("STAGE 8: COST-SENSITIVE LEARNING & THRESHOLD OPTIMIZATION (SECTION E)")
    print("=" * 70)

    # 1. Load Policy-Validation partition ONLY
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    print(f"Loaded Policy-Validation partition: {X_val.shape[0]} rows, {y_val.sum()} frauds ({y_val.mean()*100:.2f}%)")

    # 2. Load Frozen Production Calibrated Pipeline from Stage 7
    model_path = "04_calibrated_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Frozen calibrated model not found at {model_path}")
    calibrated_pipeline = joblib.load(model_path)
    print(f"Loaded frozen calibrated pipeline from {model_path}")

    # Generate calibrated probabilities on Policy-Validation
    probs_val = calibrated_pipeline.predict_proba(X_val)[:, 1]
    amounts_val = X_val["transaction_amount"].to_numpy()

    # Load business cost configuration
    cost_cfg = load_business_cost_config("config/business_costs.yaml")
    scenarios = cost_cfg["scenarios"]

    # -------------------------------------------------------------
    # 3. Baseline & Two-Threshold Optimization on Primary Scenario B
    # -------------------------------------------------------------
    primary_scenario_key = "scenario_b_balanced"
    primary_cfg = scenarios[primary_scenario_key]
    print(f"\nRunning Optimization on Primary Scenario: {primary_cfg['name']}...")

    # Policy 1: Default Binary Threshold t = 0.50
    default_policy = evaluate_policy_costs(
        y_true=y_val.to_numpy(),
        probs=probs_val,
        amounts=amounts_val,
        review_threshold=0.50,
        block_threshold=0.50,
        scenario_cfg=primary_cfg,
    )
    default_policy["policy_name"] = "Default Binary (t=0.50)"

    # Policy 2: Cost-Optimized Binary Threshold
    binary_grid_df, best_binary = grid_search_thresholds(
        y_true=y_val.to_numpy(),
        probs=probs_val,
        amounts=amounts_val,
        scenario_cfg=primary_cfg,
        step=0.01,
        mode="binary",
    )
    best_binary["policy_name"] = f"Optimal Binary (t={best_binary['block_threshold']:.2f})"

    # Policy 3: Two-Threshold Operational Policy (APPROVE / REVIEW / BLOCK)
    two_th_grid_df, best_two_th = grid_search_thresholds(
        y_true=y_val.to_numpy(),
        probs=probs_val,
        amounts=amounts_val,
        scenario_cfg=primary_cfg,
        step=0.02,
        mode="two_threshold",
    )
    best_two_th["policy_name"] = f"Two-Threshold (Rev={best_two_th['review_threshold']:.2f}, Blk={best_two_th['block_threshold']:.2f})"

    # Policy 4: Conservative Low-Friction Binary
    conservative_policy = evaluate_policy_costs(
        y_true=y_val.to_numpy(),
        probs=probs_val,
        amounts=amounts_val,
        review_threshold=0.20,
        block_threshold=0.20,
        scenario_cfg=primary_cfg,
    )
    conservative_policy["policy_name"] = "Triage Binary (t=0.20)"

    comparison_rows = [default_policy, conservative_policy, best_binary, best_two_th]
    comp_df = pd.DataFrame(comparison_rows)

    # Compute savings vs Default
    def_cost = default_policy["total_cost"]
    comp_df["cost_savings_vs_default"] = def_cost - comp_df["total_cost"]
    comp_df["savings_pct"] = (comp_df["cost_savings_vs_default"] / def_cost) * 100.0

    print("\nPolicy Cost Comparison on Primary Scenario B (Policy-Validation n=600):")
    display_cols = [
        "policy_name",
        "total_cost",
        "cost_per_1k",
        "fraud_capture_rate",
        "review_workload_pct",
        "block_rate_pct",
        "fraud_approved",
        "legit_blocked",
        "cost_savings_vs_default",
        "savings_pct",
    ]
    print(comp_df[display_cols].to_string(index=False))

    # -------------------------------------------------------------
    # 4. Multi-Scenario Comparison (Scenarios A, B, C)
    # -------------------------------------------------------------
    print("\nEvaluating Across All Three Business Scenarios (A, B, C)...")
    scenario_records = []

    for s_key, s_cfg in scenarios.items():
        # Evaluate default
        p_def = evaluate_policy_costs(
            y_val.to_numpy(), probs_val, amounts_val, 0.50, 0.50, s_cfg
        )
        # Evaluate optimal 2-threshold
        _, p_opt = grid_search_thresholds(
            y_val.to_numpy(), probs_val, amounts_val, s_cfg, step=0.02, mode="two_threshold"
        )
        scenario_records.append({
            "scenario": s_cfg["name"],
            "default_cost_1k": p_def["cost_per_1k"],
            "optimal_cost_1k": p_opt["cost_per_1k"],
            "optimal_review_th": p_opt["review_threshold"],
            "optimal_block_th": p_opt["block_threshold"],
            "optimal_fraud_capture": p_opt["fraud_capture_rate"] * 100.0,
            "optimal_review_pct": p_opt["review_workload_pct"],
            "cost_savings_pct": ((p_def["total_cost"] - p_opt["total_cost"]) / p_def["total_cost"]) * 100.0,
        })

    scenario_summary_df = pd.DataFrame(scenario_records)
    print("\nScenario Summary:")
    print(scenario_summary_df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. Candidate Architecture Cost Check (RF vs LR vs HGB)
    # -------------------------------------------------------------
    print("\nComparing Cost Performance Across Candidate Architectures on Policy-Validation...")
    model_cost_records = []
    candidates = [
        ("RF_Tuned", "04_calibrated_model.joblib"),
        ("LR_Tuned", "reports/intermediate/candidate_lr_pipeline.joblib"),
        ("HGB_Tuned", "reports/intermediate/candidate_hgb_pipeline.joblib"),
    ]
    for model_name, model_file in candidates:
        if os.path.exists(model_file):
            m = joblib.load(model_file)
            p = m.predict_proba(X_val)[:, 1]
            p_res = evaluate_policy_costs(
                y_true=y_val.to_numpy(),
                probs=p,
                amounts=amounts_val,
                review_threshold=best_two_th["review_threshold"],
                block_threshold=best_two_th["block_threshold"],
                scenario_cfg=primary_cfg,
            )
            # Also find optimal for that specific model
            _, p_spec = grid_search_thresholds(
                y_true=y_val.to_numpy(),
                probs=p,
                amounts=amounts_val,
                scenario_cfg=primary_cfg,
                step=0.02,
                mode="two_threshold",
            )
            model_cost_records.append({
                "model": model_name,
                "cost_at_selected_policy_1k": p_res["cost_per_1k"],
                "optimal_cost_1k": p_spec["cost_per_1k"],
                "opt_review_th": p_spec["review_threshold"],
                "opt_block_th": p_spec["block_threshold"],
                "fraud_capture_pct": p_spec["fraud_capture_rate"] * 100.0,
                "review_pct": p_spec["review_workload_pct"],
            })

    candidate_cost_df = pd.DataFrame(model_cost_records)
    print(candidate_cost_df.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Sensitivity Analysis
    # -------------------------------------------------------------
    print("\nRunning Multidimensional Parameter Sensitivity Analysis...")
    sensitivity_df = run_cost_sensitivity_analysis(
        y_true=y_val.to_numpy(),
        probs=probs_val,
        amounts=amounts_val,
        base_scenario_cfg=primary_cfg,
        sensitivity_cfg=cost_cfg["sensitivity_grid"],
    )
    print(f"Generated sensitivity grid with {len(sensitivity_df)} parameter evaluations.")

    # -------------------------------------------------------------
    # 7. Persist Tables and Required CSVs
    # -------------------------------------------------------------
    os.makedirs("reports/tables", exist_ok=True)

    # 05_threshold_cost_analysis.csv in root as explicitly required
    two_th_grid_df.to_csv("05_threshold_cost_analysis.csv", index=False)
    print("Saved 05_threshold_cost_analysis.csv to root directory.")

    comp_df.to_csv("reports/tables/business_cost_comparison.csv", index=False)
    scenario_summary_df.to_csv("reports/tables/business_scenario_summary.csv", index=False)
    candidate_cost_df.to_csv("reports/tables/model_cost_architecture_comparison.csv", index=False)
    sensitivity_df.to_csv("reports/tables/cost_sensitivity.csv", index=False)
    print("Saved all comparison and sensitivity tables to reports/tables/.")

    # -------------------------------------------------------------
    # 8. Render Visualizations
    # -------------------------------------------------------------
    print("\nGenerating publication-quality cost visualizations (300 DPI)...")
    cost_figs = plot_threshold_cost_surfaces(
        binary_grid_df=binary_grid_df,
        two_th_grid_df=two_th_grid_df,
        output_dir="reports/figures",
    )
    for name, path in cost_figs.items():
        print(f"  Generated [{name}]: {path}")

    sens_fig = plot_sensitivity_heatmaps(
        sensitivity_df=sensitivity_df,
        output_dir="reports/figures",
    )
    print(f"  Generated [sensitivity]: {sens_fig}")

    print("\nStage 8 cost-sensitive learning and threshold optimization completed successfully!")


if __name__ == "__main__":
    run_stage_8()
