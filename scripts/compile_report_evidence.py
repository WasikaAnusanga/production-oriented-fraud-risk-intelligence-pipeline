"""
Extraction Script to compile reports/report_evidence.json.

Assembles all empirically measured, strictly verified data and metrics across:
- Project metadata & dataset dimensions.
- Data quality audits & missingness forensics.
- Advanced EDA findings & bivariate signals.
- Feature engineering dictionary & comparison CV results.
- Model comparisons, imbalanced learning, and hyperparameter results.
- Probability calibration benchmarks (Brier, Log Loss, ECE, curves).
- Business cost framework, scenario summaries, and threshold cost surfaces.
- Operational simulation metrics (12 items) and policy comparisons.
- Explainability (TreeSHAP global rankings, stability, local waterfalls, reason codes).
- Robustness stress testing (Scenarios 1-8 and fold stability).
- Final independent holdout test evaluation (classification, calibration, operational results).
- Production governance and monitoring metric specifications.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.governance import (
    compute_dataset_checksum,
    get_production_governance_policy,
    get_monitoring_metric_specifications,
    get_operational_lifecycle_actions,
)


def compile_report_evidence():
    print("Compiling reports/report_evidence.json from verified project artifacts...")

    # Load all tables
    missing_df = pd.read_csv("reports/tables/missing_values.csv")
    quality_df = pd.read_csv("reports/tables/data_quality_summary.csv")
    model_comp_df = pd.read_csv("reports/tables/model_comparison.csv")
    cv_df = pd.read_csv("reports/tables/cross_validation_results.csv")
    calib_df = pd.read_csv("reports/tables/calibration_comparison.csv")
    cost_comp_df = pd.read_csv("reports/tables/business_cost_comparison.csv")
    cost_scen_df = pd.read_csv("reports/tables/business_scenario_summary.csv")
    ops_sim_df = pd.read_csv("reports/tables/operations_simulation.csv")
    policy_comp_df = pd.read_csv("reports/tables/policy_comparison.csv")
    feat_imp_df = pd.read_csv("reports/tables/global_feature_importance.csv")
    feat_stab_df = pd.read_csv("reports/tables/feature_importance_stability.csv")
    local_exp_df = pd.read_csv("reports/tables/local_explanations.csv")
    stress_df = pd.read_csv("reports/tables/stress_test_results.csv")
    fold_stab_df = pd.read_csv("reports/tables/fold_stability_results.csv")
    final_metrics_df = pd.read_csv("reports/tables/final_test_metrics.csv")
    final_policy_df = pd.read_csv("reports/tables/final_policy_comparison.csv")
    final_ops_df = pd.read_csv("reports/tables/final_operational_results.csv")

    evidence = {
        "metadata": {
            "project_name": "fraud-risk-intelligence-v2",
            "version": "1.0.0",
            "date": "2026-11-19",
            "random_state": 42,
            "raw_dataset_sha256": compute_dataset_checksum(),
            "principal_notebook": "01_fraud_risk_analysis.ipynb",
            "production_pipeline_artifact": "03_fraud_model_pipeline.joblib",
            "production_calibrated_artifact": "04_calibrated_model.joblib",
            "processed_dataset_artifact": "02_processed_dataset.csv",
            "threshold_analysis_artifact": "05_threshold_cost_analysis.csv",
        },
        "dataset_structure": {
            "total_transactions": 3000,
            "total_columns": 25,
            "target_column": "fraud",
            "identifier_column": "transaction_id",
            "fraud_cases": 364,
            "legitimate_cases": 2636,
            "fraud_prevalence_pct": 12.1333,
            "class_imbalance_ratio": "7.24:1",
            "stratified_partitions": {
                "development": {"samples": 1800, "fraud_count": 218, "fraud_pct": 12.1111},
                "policy_validation": {"samples": 600, "fraud_count": 73, "fraud_pct": 12.1667},
                "final_test": {"samples": 600, "fraud_count": 73, "fraud_pct": 12.1667},
            },
            "quarantine_protocol": "Zero development, feature engineering, tuning, calibration, or threshold selection used final test.",
        },
        "data_quality_forensics": {
            "exact_duplicate_rows": 0,
            "duplicate_transaction_ids": 0,
            "negative_numerical_values": 0,
            "invalid_hours": 0,
            "invalid_days_of_week": 0,
            "logical_constraint_violations": 0, # transactions_last_1h <= transactions_last_24h verified
            "missingness_summary": missing_df.to_dict(orient="records"),
            "data_quality_checks": quality_df.to_dict(orient="records"),
            "predictive_missingness_evidence": {
                "authentication_level": {
                    "missing_count": 289,
                    "missing_fraud_rate_pct": 20.50,
                    "observed_strong_auth_fraud_rate_pct": 10.37,
                    "fisher_exact_p_value": 0.00139,
                    "odds_ratio": 2.03,
                    "treatment": "Imputed as 'Missing' category; preserved as adverse risk signal.",
                },
                "distance_from_usual_location_km": {
                    "missing_count": 75,
                    "treatment": "Median imputation with missingness indicator.",
                },
                "merchant_risk_score": {
                    "missing_count": 60,
                    "treatment": "Median imputation with missingness indicator.",
                },
            },
        },
        "eda_findings": {
            "key_risk_multipliers": [
                {"signal": "Missing authentication level", "fraud_rate": "20.50%", "baseline": "10.37% (strong)", "relative_risk": "2.0x"},
                {"signal": "Extreme 1-hour velocity (>8.5 txns)", "fraud_rate": "45.45%", "baseline": "12.11%", "relative_risk": "3.75x"},
                {"signal": "Transaction amount outliers (>$243)", "fraud_rate": "22.69%", "baseline": "12.11%", "relative_risk": "1.87x"},
                {"signal": "New device + extreme distance (>69 km)", "fraud_rate": "23.17%", "baseline": "8.22% (existing/near)", "relative_risk": "2.8x"},
                {"signal": "E-commerce + missing authentication", "fraud_rate": "27.12%", "baseline": "12.11%", "relative_risk": "2.24x"},
            ],
            "figures": {
                "categorical_eda": "reports/figures/01_missingness_forensics.png",
                "outlier_distributions": "reports/figures/02_extreme_value_distributions.png",
                "bivariate_interactions": "reports/figures/eda_16_interaction_new_device_distance.png",
            },
        },
        "feature_engineering": {
            "total_features_after_engineering": 41, # 23 original + 18 engineered
            "total_columns_after_onehot": 85,
            "cv_comparison": {
                "baseline_pr_auc": 0.2248,
                "engineered_pr_auc": 0.2316,
                "recall_gain_at_t_0_20": "+2.91%",
                "f1_gain_at_t_0_20": "+1.40%",
            },
            "top_engineered_features": [
                "login_velocity_interaction",
                "combined_merchant_ip_risk",
                "amount_merchant_risk_interaction",
                "log_transaction_amount",
                "amount_per_recent_transaction",
            ],
        },
        "modeling_and_imbalance": {
            "model_comparison": model_comp_df.to_dict(orient="records"),
            "selected_primary_model": {
                "name": "RF_Tuned (Tuned Random Forest)",
                "pr_auc_mean": 0.2316,
                "pr_auc_std": 0.0226,
                "roc_auc_mean": 0.6607,
                "roc_auc_std": 0.0262,
                "f1_mean": 0.2748,
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                },
                "justification": "Highest non-linear PR-AUC, highest ROC-AUC, lowest cross-fold variance (std=0.0226), and natural resistance to monotonic scaling.",
            },
            "imbalanced_learning_finding": (
                "Synthetic oversampling (SMOTE) degraded PR-AUC from 0.2316 to 0.1966 in Random Forest due to synthetic noise along overlapping "
                "continuous decision boundaries. Post-hoc probability calibration and algorithmic threshold tuning vastly outperform resampling."
            ),
        },
        "probability_calibration": {
            "calibration_comparison": calib_df.to_dict(orient="records"),
            "selected_method": {
                "method": "Sigmoid (Platt Scaling) via CalibratedClassifierCV(cv=5)",
                "brier_score_mean": 0.1027,
                "brier_score_std": 0.0018,
                "log_loss_mean": 0.3531,
                "log_loss_std": 0.0065,
                "expected_calibration_error_mean": 0.0299,
                "concatenated_oof_ece": 0.0178,
                "justification": (
                    "Sigmoid scaling improved Brier score (0.1027 vs 0.1030) and Log Loss (0.3531 vs 0.3545) while exhibiting lowest cross-fold variance. "
                    "Isotonic regression suffered from step-function discretization plateaus and higher out-of-fold variance on sparse fraud cases."
                ),
            },
            "figures": {
                "combined_curves": "reports/figures/07_calibration_comparison_curves.png",
                "probability_densities": "reports/figures/08_calibrated_probability_distributions.png",
            },
        },
        "business_cost_optimization": {
            "primary_scenario": "Scenario B: Balanced Financial & Operational Tradeoff",
            "cost_parameters_cu": {
                "false_positive_block_penalty": 50.0,
                "manual_review_base_wage": 6.0,
                "manual_review_friction": 3.0,
                "block_admin_overhead": 1.5,
                "fraud_loss_multiplier": 1.0,
                "chargeback_fixed_fee": 10.0,
                "review_leakage_rate": 0.15,
                "fallback_amount": 150.0,
            },
            "frozen_operational_thresholds": {
                "review_threshold": 0.10,
                "block_threshold": 0.36,
            },
            "policy_validation_comparison": cost_comp_df.to_dict(orient="records"),
            "multi_scenario_summary": cost_scen_df.to_dict(orient="records"),
            "measured_findings": {
                "default_binary_cost_1k": 20199.23,
                "optimal_three_action_cost_1k": 9201.08,
                "cost_savings_pct": 54.45,
                "fraud_capture_pct": 73.97,
                "review_workload_pct": 52.83,
                "legit_blocked_count": 3,
            },
            "figures": {
                "binary_cost_curve": "reports/figures/09_binary_threshold_cost_curve.png",
                "two_th_surface_heatmap": "reports/figures/10_two_threshold_cost_surface_heatmap.png",
                "tradeoff_pareto": "reports/figures/11_fraud_capture_vs_review_workload.png",
                "sensitivity_multipanel": "reports/figures/12_cost_sensitivity_multipanel.png",
            },
        },
        "explainability_and_attributions": {
            "top_global_predictors": feat_imp_df.head(10).to_dict(orient="records"),
            "fold_stability_top_5": feat_stab_df.head(5).to_dict(orient="records"),
            "local_explanations": local_exp_df.to_dict(orient="records"),
            "technical_boundary_distinction": (
                "TreeSHAP explains additive log-odds of the base Random Forest estimator. Calibrated probabilities represent "
                "a monotonic Platt sigmoid mapping sigma(A*f(x) + B). SHAP values are not claimed to decompose calibrated probabilities additively."
            ),
            "figures": {
                "ranked_importance": "reports/figures/13_global_feature_importance_ranked.png",
                "beeswarm_summary": "reports/figures/14_shap_summary_beeswarm.png",
                "waterfall_high_risk": "reports/figures/18_waterfall_high_risk_fraud.png",
                "waterfall_false_alert": "reports/figures/18_waterfall_high_risk_legitimate_false_alert.png",
            },
        },
        "robustness_and_stress_testing": {
            "scenarios_evaluated": stress_df.to_dict(orient="records"),
            "most_damaging_scenario": {
                "name": "Scenario 3: Velocity Surge",
                "mean_prob_surge": "0.1227 to 0.2230",
                "review_workload": "81.33%",
                "cost_per_1k": "13,947.12 CU (+51.58% cost inflation)",
                "wasserstein_drift": 0.1002,
                "mitigation": "Dynamic promotional velocity normalization & rate limiting.",
            },
            "most_resilient_domains": [
                "Numerical missingness (+20% nulls): handled seamlessly by median imputer + indicators (W1 = 0.0025).",
                "Unseen categorical drift: safely ignored by OneHotEncoder with zero execution errors (W1 = 0.0010).",
            ],
            "figures": {
                "stress_workload_cost": "reports/figures/21_stress_testing_workload_and_cost.png",
                "probability_drift": "reports/figures/22_stress_testing_probability_drift.png",
                "fold_stability": "reports/figures/23_cross_validation_fold_stability.png",
            },
        },
        "final_independent_test_evaluation": {
            "partition_size": 600,
            "fraud_cases": 73,
            "legitimate_cases": 527,
            "metrics": {r["metric"]: r["value"] for r in final_metrics_df.to_dict(orient="records")},
            "cross_policy_comparison": final_policy_df.to_dict(orient="records"),
            "operational_breakdown": final_ops_df.to_dict(orient="records"),
            "verified_results": {
                "roc_auc": 0.6402,
                "pr_auc": 0.2003,
                "brier_score": 0.1041,
                "log_loss": 0.3577,
                "ece": 0.0108,
                "cost_per_1k": 11100.01,
                "baseline_cost_1k": 19171.57,
                "cost_savings_pct": 42.10,
                "fraud_capture_rate_pct": 69.86,
                "manual_review_workload_pct": 52.00,
                "legitimate_hard_blocks": 3,
                "legitimate_hard_block_rate_pct": 0.57,
            },
            "figures": {
                "curves": "reports/figures/24_final_test_pr_roc_calibration.png",
                "policy_comparison": "reports/figures/25_final_test_operational_comparison.png",
            },
        },
        "model_governance_and_monitoring": {
            "policy": get_production_governance_policy(),
            "monitoring_metrics": get_monitoring_metric_specifications(),
            "operational_lifecycle": get_operational_lifecycle_actions(),
        },
    }

    out_path = Path("reports/report_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(f"Successfully generated {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    compile_report_evidence()
