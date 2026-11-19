"""
Execution Script for Stage 10: Fraud Operations Simulation & Policy Comparison (Section I).

Runs:
1. Production decision engine batch inference simulation over Policy-Validation partition.
2. 12-item operational breakdown under the frozen 3-tier policy (t_rev=0.10, t_blk=0.36).
3. Cross-policy comparison against Standard Binary (t=0.50) and Cost-Optimized Binary (t=0.19).
4. Exports reports/tables/operations_simulation.csv and reports/tables/policy_comparison.csv.
5. Renders publication dashboards: 19_operational_decision_distribution.png and 20_cross_policy_operational_comparison.png.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config.settings import RANDOM_STATE
from src.data_splitting import get_partition_data
from src.fraud_decision_engine import (
    FraudDecisionEngine,
    run_operational_simulation,
    plot_operational_dashboards,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
)
from src.cost_optimization import load_business_cost_config


def run_stage_10():
    print("=" * 70)
    print("STAGE 10: FRAUD OPERATIONS SIMULATION (SECTION I)")
    print("=" * 70)

    # 1. Load Policy-Validation cohort (n=600)
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    amounts_val = X_val["transaction_amount"].to_numpy()
    print(f"Loaded Policy-Validation cohort: {len(X_val)} transactions, {y_val.sum()} frauds ({y_val.mean()*100:.2f}%)")

    # 2. Initialize FraudDecisionEngine wrapping frozen calibrated pipeline
    engine = FraudDecisionEngine(
        model_path="04_calibrated_model.joblib",
        review_threshold=DEFAULT_REVIEW_THRESHOLD,
        block_threshold=DEFAULT_BLOCK_THRESHOLD,
    )
    print(f"Initialized FraudDecisionEngine (t_review={engine.review_threshold}, t_block={engine.block_threshold})")

    # 3. Test predict_transaction_risk batch inference
    print("\nExecuting predict_transaction_risk batch test on first 5 transactions...")
    sample_df = X_val.head(5).copy()
    sample_df["transaction_id"] = ids_val.head(5).values
    sample_preds = engine.predict_transaction_risk(sample_df, include_explanations=True)
    print(sample_preds[["transaction_id", "fraud_probability", "operational_decision", "risk_level"]].to_string(index=False))

    # 4. Run formal operational simulation and cross-policy benchmark
    print("\nRunning Operational Simulation over Policy-Validation Cohort (n=600)...")
    cost_cfg = load_business_cost_config("config/business_costs.yaml")
    primary_scenario_cfg = cost_cfg["scenarios"]["scenario_b_balanced"]

    sim_df, policy_comp_df = run_operational_simulation(
        calibrated_pipeline=engine.model,
        X_val=X_val,
        y_val=y_val,
        amounts_val=amounts_val,
        scenario_cfg=primary_scenario_cfg,
        review_threshold=engine.review_threshold,
        block_threshold=engine.block_threshold,
    )

    print("\nDetailed 12-Item Operational Simulation Breakdown:")
    print(sim_df.to_string(index=False))

    print("\nCross-Policy Comparison:")
    disp_cols = [
        "policy_name",
        "review_th",
        "block_th",
        "total_cost",
        "cost_per_1k",
        "fraud_capture_rate",
        "review_workload_pct",
        "legit_blocked",
        "cost_savings_pct",
    ]
    print(policy_comp_df[disp_cols].to_string(index=False))

    # 5. Export Tables
    os.makedirs("reports/tables", exist_ok=True)
    sim_df.to_csv("reports/tables/operations_simulation.csv", index=False)
    policy_comp_df.to_csv("reports/tables/policy_comparison.csv", index=False)
    print("\nSaved operational simulation tables to reports/tables/.")

    # 6. Render Operational Dashboards (300 DPI)
    print("\nGenerating operational dashboards...")
    dash_figs = plot_operational_dashboards(
        simulation_df=sim_df,
        policy_comp_df=policy_comp_df,
        output_dir="reports/figures",
    )
    for name, p in dash_figs.items():
        print(f"  Generated [{name}]: {p}")

    print("\nStage 10 Fraud Operations Simulation completed successfully!")


if __name__ == "__main__":
    run_stage_10()
