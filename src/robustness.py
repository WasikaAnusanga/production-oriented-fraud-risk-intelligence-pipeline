"""
Robustness and Stress Testing Module for Fraud Risk Intelligence Pipeline.

Evaluates pipeline resilience under shifting production conditions without retraining:
- Monetary inflation (+25%, +50%).
- Velocity surges (1h, 24h) maintaining logical constraints.
- Channel shifts toward digital channels (E-commerce, Mobile App).
- Numerical and categorical missingness spikes.
- Extreme outlier telemetry (amounts, distances).
- Unseen categorical combinations and corruptions.
- Cross-validation fold performance variability.

Tracks:
- Mean probability, probability drift (Wasserstein / KS distance).
- Action distributions (APPROVE, MANUAL REVIEW, BLOCK).
- Manual review workload and false hard blocks.
- Conditional performance metrics (ROC-AUC, PR-AUC, Recall, Precision, Brier Score).
- Realized business costs (Scenario B Balanced Baseline).
"""

from typing import Dict, Any, List, Tuple, Optional
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    recall_score,
    precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config.settings import RANDOM_STATE, NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS
from src.cost_optimization import load_business_cost_config, evaluate_policy_costs
from src.fraud_decision_engine import DEFAULT_REVIEW_THRESHOLD, DEFAULT_BLOCK_THRESHOLD


def evaluate_cohort_performance(
    calibrated_model,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    scenario_cfg: Dict[str, Any],
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluates risk distribution, operational routing, and conditional performance
    for a stressed or shifted cohort.
    """
    n_samples = len(X)
    probs = calibrated_model.predict_proba(X)[:, 1]
    mean_prob = float(np.mean(probs))
    median_prob = float(np.median(probs))
    std_prob = float(np.std(probs))

    # Action routing
    actions = np.full(n_samples, "APPROVE", dtype=object)
    actions[(probs >= review_threshold) & (probs < block_threshold)] = "MANUAL REVIEW"
    actions[probs >= block_threshold] = "BLOCK"

    pct_approve = float(np.sum(actions == "APPROVE") / n_samples * 100.0)
    pct_review = float(np.sum(actions == "MANUAL REVIEW") / n_samples * 100.0)
    pct_block = float(np.sum(actions == "BLOCK") / n_samples * 100.0)

    amounts = X["transaction_amount"].to_numpy()

    # Cost and performance evaluation
    if y is not None:
        y_arr = y.to_numpy().astype(int)
        policy_res = evaluate_policy_costs(
            y_true=y_arr,
            probs=probs,
            amounts=amounts,
            review_threshold=review_threshold,
            block_threshold=block_threshold,
            scenario_cfg=scenario_cfg,
        )

        roc_auc = float(roc_auc_score(y_arr, probs))
        pr_auc = float(average_precision_score(y_arr, probs))
        brier = float(brier_score_loss(y_arr, probs))

        # Binary flag (REVIEW or BLOCK)
        pred_pos = (actions != "APPROVE").astype(int)
        recall = float(recall_score(y_arr, pred_pos, zero_division=0))
        precision = float(precision_score(y_arr, pred_pos, zero_division=0))

        fraud_capture = policy_res["fraud_capture_rate"] * 100.0
        legit_blocked = policy_res["legit_blocked"]
        total_cost = policy_res["total_cost"]
        cost_per_1k = policy_res["cost_per_1k"]
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        brier = np.nan
        recall = np.nan
        precision = np.nan
        fraud_capture = np.nan
        legit_blocked = np.nan
        total_cost = np.nan
        cost_per_1k = np.nan

    return {
        "n_samples": n_samples,
        "mean_probability": mean_prob,
        "median_probability": median_prob,
        "std_probability": std_prob,
        "pct_approve": pct_approve,
        "pct_review": pct_review,
        "pct_block": pct_block,
        "review_workload_pct": pct_review,
        "fraud_capture_pct": fraud_capture,
        "legit_blocked_count": legit_blocked,
        "total_cost": total_cost,
        "cost_per_1k": cost_per_1k,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "recall": recall,
        "precision": precision,
        "probabilities": probs,
    }


def generate_stress_scenarios(
    X_base: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """
    Generates 8 reproducible stress-testing datasets from the base DataFrame:
    1. Scenario 1: +25% transaction_amount
    2. Scenario 2: +50% transaction_amount
    3. Scenario 3: Velocity surge (1h, 24h) maintaining constraints
    4. Scenario 4: Channel shift to E-commerce & Mobile App
    5. Scenario 5: Additional numerical missingness (+20% nulls in 4 fields)
    6. Scenario 6: Missing authentication level spike (+30% nulls)
    7. Scenario 7: Extreme amounts & distances (heavy-tail shocks)
    8. Scenario 8: Uncommon & unseen categorical corruptions
    """
    rng = np.random.RandomState(random_state)
    scenarios = {}

    # Scenario 1: +25% transaction_amount
    s1 = X_base.copy()
    mask_amt = s1["transaction_amount"].notnull()
    s1.loc[mask_amt, "transaction_amount"] = s1.loc[mask_amt, "transaction_amount"] * 1.25
    scenarios["Scenario 1: Amount +25%"] = s1

    # Scenario 2: +50% transaction_amount
    s2 = X_base.copy()
    mask_amt = s2["transaction_amount"].notnull()
    s2.loc[mask_amt, "transaction_amount"] = s2.loc[mask_amt, "transaction_amount"] * 1.50
    scenarios["Scenario 2: Amount +50%"] = s2

    # Scenario 3: Consistent velocity surge maintaining constraints
    s3 = X_base.copy()
    added_1h = rng.randint(2, 6, size=len(s3))
    added_24h = added_1h + rng.randint(2, 8, size=len(s3))
    s3["transactions_last_1h"] = s3["transactions_last_1h"] + added_1h
    s3["transactions_last_24h"] = s3["transactions_last_24h"] + added_24h
    # Ensure logical constraint: 24h >= 1h
    s3["transactions_last_24h"] = np.maximum(s3["transactions_last_24h"], s3["transactions_last_1h"])
    scenarios["Scenario 3: Velocity Surge"] = s3

    # Scenario 4: Digital channel shift (80% E-commerce, 20% Mobile App)
    s4 = X_base.copy()
    digital_channels = rng.choice(["E-commerce", "Mobile App"], size=len(s4), p=[0.75, 0.25])
    s4["channel"] = digital_channels
    scenarios["Scenario 4: Digital Channel Shift"] = s4

    # Scenario 5: Numerical missingness injection (20% missing in 4 numeric columns)
    s5 = X_base.copy()
    for col in ["merchant_risk_score", "ip_risk_score", "distance_from_usual_location_km", "login_to_transaction_minutes"]:
        mask_drop = rng.rand(len(s5)) < 0.20
        s5.loc[mask_drop, col] = np.nan
    scenarios["Scenario 5: Numerical Missingness Spike"] = s5

    # Scenario 6: Missing authentication level spike (additional 30% missing)
    s6 = X_base.copy()
    mask_auth_drop = rng.rand(len(s6)) < 0.30
    s6.loc[mask_auth_drop, "authentication_level"] = np.nan
    scenarios["Scenario 6: Missing Auth Spike (+30%)"] = s6

    # Scenario 7: Extreme amounts & distances (outlier injection on top 10% volume)
    s7 = X_base.copy()
    mask_outliers = rng.rand(len(s7)) < 0.10
    s7.loc[mask_outliers, "transaction_amount"] = s7.loc[mask_outliers, "transaction_amount"] * 3.5
    s7.loc[mask_outliers, "distance_from_usual_location_km"] = s7.loc[mask_outliers, "distance_from_usual_location_km"] * 4.0 + 150.0
    scenarios["Scenario 7: Extreme Telemetry Outliers"] = s7

    # Scenario 8: Uncommon & unseen categorical corruptions
    s8 = X_base.copy()
    mask_unseen = rng.rand(len(s8)) < 0.15
    s8.loc[mask_unseen, "channel"] = "UNSEEN_VR_COMMERCE"
    s8.loc[mask_unseen, "device_type"] = "QUANTUM_HEADSET"
    s8.loc[mask_unseen, "customer_tier"] = "ULTRA_TIER_99"
    s8.loc[mask_unseen, "country"] = "ZZ_ATLANTIS"
    scenarios["Scenario 8: Unseen Categorical Drift"] = s8

    return scenarios


def run_comprehensive_stress_testing(
    calibrated_model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scenario_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Executes baseline and all 8 stress scenarios on Policy-Validation cohort.
    Calculates Wasserstein distance relative to baseline probability distribution.
    """
    base_res = evaluate_cohort_performance(
        calibrated_model=calibrated_model,
        X=X_val,
        y=y_val,
        scenario_cfg=scenario_cfg,
    )
    base_probs = base_res["probabilities"]

    records = [{
        "scenario": "Baseline (Policy-Validation)",
        "mean_probability": base_res["mean_probability"],
        "pct_approve": base_res["pct_approve"],
        "pct_review": base_res["pct_review"],
        "pct_block": base_res["pct_block"],
        "fraud_capture_pct": base_res["fraud_capture_pct"],
        "legit_blocked_count": base_res["legit_blocked_count"],
        "cost_per_1k": base_res["cost_per_1k"],
        "roc_auc": base_res["roc_auc"],
        "pr_auc": base_res["pr_auc"],
        "brier_score": base_res["brier_score"],
        "recall": base_res["recall"],
        "precision": base_res["precision"],
        "wasserstein_drift": 0.0,
    }]

    stress_dict = generate_stress_scenarios(X_val)

    for s_name, X_stressed in stress_dict.items():
        res = evaluate_cohort_performance(
            calibrated_model=calibrated_model,
            X=X_stressed,
            y=y_val,  # conditional benchmark with original ground truth
            scenario_cfg=scenario_cfg,
        )
        w_dist = float(wasserstein_distance(base_probs, res["probabilities"]))

        records.append({
            "scenario": s_name,
            "mean_probability": res["mean_probability"],
            "pct_approve": res["pct_approve"],
            "pct_review": res["pct_review"],
            "pct_block": res["pct_block"],
            "fraud_capture_pct": res["fraud_capture_pct"],
            "legit_blocked_count": res["legit_blocked_count"],
            "cost_per_1k": res["cost_per_1k"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "brier_score": res["brier_score"],
            "recall": res["recall"],
            "precision": res["precision"],
            "wasserstein_drift": w_dist,
        })

    return pd.DataFrame(records)


def evaluate_development_fold_stability(
    calibrated_model,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    scenario_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Scenario 9: Evaluates model performance and cost stability across the 5 internal
    calibrator folds on the development partition.
    """
    estimators = [cal_clf.estimator for cal_clf in calibrated_model.calibrated_classifiers_]
    records = []

    for idx, est in enumerate(estimators):
        probs = est.predict_proba(X_dev)[:, 1]
        roc = float(roc_auc_score(y_dev, probs))
        pr = float(average_precision_score(y_dev, probs))
        brier = float(brier_score_loss(y_dev, probs))

        amounts = X_dev["transaction_amount"].to_numpy()
        policy_res = evaluate_policy_costs(
            y_true=y_dev.to_numpy(),
            probs=probs,
            amounts=amounts,
            review_threshold=DEFAULT_REVIEW_THRESHOLD,
            block_threshold=DEFAULT_BLOCK_THRESHOLD,
            scenario_cfg=scenario_cfg,
        )

        records.append({
            "fold": idx + 1,
            "roc_auc": roc,
            "pr_auc": pr,
            "brier_score": brier,
            "fraud_capture_rate": policy_res["fraud_capture_rate"] * 100.0,
            "review_workload_pct": policy_res["review_workload_pct"],
            "legit_blocked": policy_res["legit_blocked"],
            "cost_per_1k": policy_res["cost_per_1k"],
        })

    df_folds = pd.DataFrame(records)
    
    # Append mean and std rows
    mean_row = {"fold": "Mean"}
    std_row = {"fold": "Std"}
    for col in ["roc_auc", "pr_auc", "brier_score", "fraud_capture_rate", "review_workload_pct", "legit_blocked", "cost_per_1k"]:
        mean_row[col] = df_folds[col].mean()
        std_row[col] = df_folds[col].std()

    summary_df = pd.concat([df_folds, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return summary_df


def plot_stress_test_visualizations(
    stress_results_df: pd.DataFrame,
    fold_stability_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> Dict[str, str]:
    """
    Renders publication-quality stress testing figures (300 DPI):
    1. Operational Workload and Cost Inflation under Stress Scenarios.
    2. Wasserstein Probability Distribution Drift across Scenarios.
    3. Cross-Validation Fold Performance & Cost Variability.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}

    # Figure 1: Workload & Cost Under Stress Scenarios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    scenarios = stress_results_df["scenario"].tolist()
    y_pos = np.arange(len(scenarios))

    # Left: Review Workload (%)
    colors_workload = ["#27AE60" if "Baseline" in s else "#E67E22" for s in scenarios]
    ax1.barh(y_pos, stress_results_df["pct_review"], color=colors_workload, edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(scenarios, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Manual Review Workload (% of Volume)", fontsize=10, fontweight="bold")
    ax1.set_title("Manual Review Capacity Demand Under Stress", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax1.invert_yaxis()

    # Right: Cost per 1k Txns (CU)
    colors_cost = ["#27AE60" if "Baseline" in s else "#C0392B" for s in scenarios]
    ax2.barh(y_pos, stress_results_df["cost_per_1k"], color=colors_cost, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Expected Business Cost per 1,000 Txns (CU)", fontsize=10, fontweight="bold")
    ax2.set_title("Financial Cost Exposure Under Stress Scenarios", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax2.invert_yaxis()

    plt.tight_layout()
    path_stress = os.path.join(output_dir, "21_stress_testing_workload_and_cost.png")
    fig.savefig(path_stress, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["workload_and_cost"] = path_stress

    # Figure 2: Wasserstein Probability Distribution Drift
    drift_df = stress_results_df[stress_results_df["scenario"] != "Baseline (Policy-Validation)"].sort_values(
        by="wasserstein_drift", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        drift_df["scenario"],
        drift_df["wasserstein_drift"],
        color="#8E44AD",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xlabel("Wasserstein Distance Relative to Baseline Probability Distribution", fontsize=10, fontweight="bold")
    ax.set_title("Probability Distribution Drift Under Input Stress Scenarios\n(Measuring Output Fragility Against Shocks)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    path_drift = os.path.join(output_dir, "22_stress_testing_probability_drift.png")
    fig.savefig(path_drift, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["probability_drift"] = path_drift

    # Figure 3: Fold Stability (Scenario 9)
    folds_numeric = fold_stability_df[fold_stability_df["fold"].isin([1, 2, 3, 4, 5])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(folds_numeric["fold"], folds_numeric["roc_auc"], marker="o", color="#2980B9", linewidth=2, label="ROC-AUC")
    ax1.plot(folds_numeric["fold"], folds_numeric["pr_auc"], marker="s", color="#E67E22", linewidth=2, label="PR-AUC")
    ax1.set_xlabel("Cross-Validation Fold", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Discrimination Score", fontsize=10, fontweight="bold")
    ax1.set_title("Fold Discrimination Stability (ROC-AUC & PR-AUC)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.15, 0.75)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    ax2.bar(folds_numeric["fold"], folds_numeric["cost_per_1k"], color="#27AE60", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Cross-Validation Fold", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cost per 1k Txns (CU)", fontsize=10, fontweight="bold")
    ax2.set_title("Fold Financial Cost Stability (Scenario B)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    path_folds = os.path.join(output_dir, "23_cross_validation_fold_stability.png")
    fig.savefig(path_folds, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["fold_stability"] = path_folds

    return generated_figures
