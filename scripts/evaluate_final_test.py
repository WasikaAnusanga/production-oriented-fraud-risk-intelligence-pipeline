"""
Final Independent Test Set Evaluation Script for Stage 12.

Strict Data Governance:
- Unlocks the 600-row Independent Final Test Set (expose_test_target=True) ONLY now that all modeling,
  calibration, and threshold parameters have been permanently frozen.
- Evaluates:
  1. Comprehensive Classification Metrics: Accuracy, Balanced Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
  2. Probability Calibration Metrics: Brier Score, Log Loss, Calibration Curves.
  3. Business & Operational Metrics under Scenario B (Balanced Baseline):
     Number approved/reviewed/blocked, Fraud capture rate, FN approved, FP blocked, Review workload/1k, Cost/1k.
- Compares:
  1. Selected 3-Action Policy (t_rev=0.10, t_blk=0.36)
  2. Default Binary Policy (t=0.50)
  3. Cost-Optimized Binary Policy (t=0.19)
- Exports:
  - reports/tables/final_test_metrics.csv
  - reports/tables/final_policy_comparison.csv
  - reports/tables/final_operational_results.csv
  - reports/figures/24_final_test_pr_roc_calibration.png
  - reports/figures/25_final_test_operational_comparison.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from config.settings import RANDOM_STATE, RAW_DATA_PATH
from src.data_splitting import get_partition_data
from src.cost_optimization import load_business_cost_config, evaluate_policy_costs
from src.calibration import compute_expected_calibration_error
from src.fraud_decision_engine import (
    FraudDecisionEngine,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
)


def run_final_test_evaluation():
    print("=" * 70)
    print("STAGE 12: FINAL INDEPENDENT TEST EVALUATION & PRODUCTION EXPORT")
    print("=" * 70)

    # 1. Unquarantine Independent Final Test Partition
    print("\nAccessing quarantined Independent Final Test partition...")
    X_test, y_test, ids_test = get_partition_data(partition="final_test", expose_test_target=True)
    if y_test is None:
        raise RuntimeError("Failed to unquarantine final test target.")
    
    n_test = len(X_test)
    total_fraud = int(y_test.sum())
    total_legit = n_test - total_fraud
    print(f"Loaded Independent Final Test partition: {n_test} transactions, {total_fraud} frauds ({y_test.mean()*100:.2f}%)")

    # 2. Reload Frozen Production Pipeline (04_calibrated_model.joblib) in fresh instance
    print("\nLoading frozen production model: 04_calibrated_model.joblib...")
    calibrated_model = joblib.load("04_calibrated_model.joblib")

    # Generate probabilities on test cohort
    probs_test = calibrated_model.predict_proba(X_test)[:, 1]
    amounts_test = X_test["transaction_amount"].to_numpy()

    # Load frozen business cost configuration
    cost_cfg = load_business_cost_config("config/business_costs.yaml")
    scenario_cfg = cost_cfg["scenarios"]["scenario_b_balanced"]

    # -------------------------------------------------------------
    # 3. Comprehensive Performance Metrics (Classification & Calibration)
    # -------------------------------------------------------------
    print("\nComputing comprehensive classification and calibration metrics on Final Test...")
    y_true = y_test.to_numpy().astype(int)

    # Binary classification metrics at selected review threshold t_rev = 0.10 (triage interception flag)
    pred_pos_triage = (probs_test >= DEFAULT_REVIEW_THRESHOLD).astype(int)
    pred_pos_default = (probs_test >= 0.50).astype(int)

    acc = accuracy_score(y_true, pred_pos_triage)
    b_acc = balanced_accuracy_score(y_true, pred_pos_triage)
    prec = precision_score(y_true, pred_pos_triage, zero_division=0)
    rec = recall_score(y_true, pred_pos_triage, zero_division=0)
    f1 = f1_score(y_true, pred_pos_triage, zero_division=0)

    roc_auc = roc_auc_score(y_true, probs_test)
    pr_auc = average_precision_score(y_true, probs_test)

    brier = brier_score_loss(y_true, probs_test)
    lloss = log_loss(y_true, np.clip(probs_test, 1e-15, 1 - 1e-15))
    ece, _, _, _ = compute_expected_calibration_error(y_true, probs_test, n_bins=10)

    tn, fp, fn, tp = confusion_matrix(y_true, pred_pos_triage).ravel()

    metrics_records = [
        {"metric": "Sample Size (n)", "value": n_test, "category": "Dataset"},
        {"metric": "Fraud Prevalance", "value": f"{y_test.mean()*100:.2f}%", "category": "Dataset"},
        {"metric": "ROC-AUC", "value": round(roc_auc, 4), "category": "Discrimination"},
        {"metric": "PR-AUC (Average Precision)", "value": round(pr_auc, 4), "category": "Discrimination"},
        {"metric": "Brier Score", "value": round(brier, 4), "category": "Calibration"},
        {"metric": "Log Loss (Cross-Entropy)", "value": round(lloss, 4), "category": "Calibration"},
        {"metric": "Expected Calibration Error (ECE)", "value": round(ece, 4), "category": "Calibration"},
        {"metric": "Accuracy (at t_review=0.10)", "value": round(acc, 4), "category": "Classification"},
        {"metric": "Balanced Accuracy", "value": round(b_acc, 4), "category": "Classification"},
        {"metric": "Precision (Triage Flag)", "value": round(prec, 4), "category": "Classification"},
        {"metric": "Recall (Triage Flag)", "value": round(rec, 4), "category": "Classification"},
        {"metric": "F1-Score", "value": round(f1, 4), "category": "Classification"},
        {"metric": "True Negatives (TN)", "value": int(tn), "category": "Confusion Matrix"},
        {"metric": "False Positives (FP)", "value": int(fp), "category": "Confusion Matrix"},
        {"metric": "False Negatives (FN)", "value": int(fn), "category": "Confusion Matrix"},
        {"metric": "True Positives (TP)", "value": int(tp), "category": "Confusion Matrix"},
    ]
    df_metrics = pd.DataFrame(metrics_records)
    print("\nFinal Test Metrics Summary:")
    print(df_metrics.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Cross-Policy Comparison on Final Test
    # -------------------------------------------------------------
    print("\nEvaluating Cross-Policy Financial Performance on Final Test...")

    # Policy 1: Frozen 3-Action Operational Policy (t_rev=0.10, t_blk=0.36)
    three_action_res = evaluate_policy_costs(
        y_true=y_true,
        probs=probs_test,
        amounts=amounts_test,
        review_threshold=DEFAULT_REVIEW_THRESHOLD,
        block_threshold=DEFAULT_BLOCK_THRESHOLD,
        scenario_cfg=scenario_cfg,
    )
    three_action_res["policy_name"] = "Optimized Three-Action Policy (Selected)"

    # Policy 2: Standard Binary Policy (t=0.50)
    binary_def_res = evaluate_policy_costs(
        y_true=y_true,
        probs=probs_test,
        amounts=amounts_test,
        review_threshold=0.50,
        block_threshold=0.50,
        scenario_cfg=scenario_cfg,
    )
    binary_def_res["policy_name"] = "Standard Binary (t=0.50)"

    # Policy 3: Cost-Optimized Binary Policy (t=0.19)
    binary_opt_res = evaluate_policy_costs(
        y_true=y_true,
        probs=probs_test,
        amounts=amounts_test,
        review_threshold=0.19,
        block_threshold=0.19,
        scenario_cfg=scenario_cfg,
    )
    binary_opt_res["policy_name"] = "Cost-Optimized Binary (t=0.19)"

    policies = [binary_def_res, binary_opt_res, three_action_res]
    df_policies = pd.DataFrame(policies)

    def_cost = binary_def_res["total_cost"]
    df_policies["cost_savings_vs_default"] = def_cost - df_policies["total_cost"]
    df_policies["cost_savings_pct"] = (df_policies["cost_savings_vs_default"] / def_cost) * 100.0

    disp_policy_cols = [
        "policy_name",
        "review_threshold",
        "block_threshold",
        "total_cost",
        "cost_per_1k",
        "fraud_capture_rate",
        "review_workload_pct",
        "legit_blocked",
        "fraud_approved",
        "cost_savings_pct",
    ]
    print("\nFinal Test Policy Comparison:")
    print(df_policies[disp_policy_cols].to_string(index=False))

    # -------------------------------------------------------------
    # 5. Detailed 12-Item Operational Results on Final Test
    # -------------------------------------------------------------
    sim_test_metrics = [
        {"metric_id": 1, "operational_metric": "Approved Transactions", "count_or_rate": three_action_res["n_approve"], "percentage_of_test_cohort": f"{(three_action_res['n_approve']/n_test)*100:.2f}%"},
        {"metric_id": 2, "operational_metric": "Transactions Routed to Manual Review", "count_or_rate": three_action_res["n_review"], "percentage_of_test_cohort": f"{(three_action_res['n_review']/n_test)*100:.2f}%"},
        {"metric_id": 3, "operational_metric": "Transactions Automatically Blocked", "count_or_rate": three_action_res["n_block"], "percentage_of_test_cohort": f"{(three_action_res['n_block']/n_test)*100:.2f}%"},
        {"metric_id": 4, "operational_metric": "Fraudulent Transactions Approved (FN Leakage)", "count_or_rate": three_action_res["fraud_approved"], "percentage_of_test_cohort": f"{(three_action_res['fraud_approved']/total_fraud)*100:.2f}% of test fraud"},
        {"metric_id": 5, "operational_metric": "Fraudulent Transactions Sent to Review", "count_or_rate": three_action_res["fraud_reviewed"], "percentage_of_test_cohort": f"{(three_action_res['fraud_reviewed']/total_fraud)*100:.2f}% of test fraud"},
        {"metric_id": 6, "operational_metric": "Fraudulent Transactions Automatically Blocked", "count_or_rate": three_action_res["fraud_blocked"], "percentage_of_test_cohort": f"{(three_action_res['fraud_blocked']/total_fraud)*100:.2f}% of test fraud"},
        {"metric_id": 7, "operational_metric": "Legitimate Transactions Unnecessarily Reviewed", "count_or_rate": three_action_res["legit_reviewed"], "percentage_of_test_cohort": f"{(three_action_res['legit_reviewed']/total_legit)*100:.2f}% of test legit"},
        {"metric_id": 8, "operational_metric": "Legitimate Transactions Incorrectly Blocked (FP Insult)", "count_or_rate": three_action_res["legit_blocked"], "percentage_of_test_cohort": f"{(three_action_res['legit_blocked']/total_legit)*100:.2f}% of test legit"},
        {"metric_id": 9, "operational_metric": "Overall Fraud Capture Rate (Review + Block)", "count_or_rate": f"{three_action_res['fraud_capture_rate']*100:.2f}%", "percentage_of_test_cohort": f"{three_action_res['fraud_reviewed'] + three_action_res['fraud_blocked']} / {total_fraud} frauds"},
        {"metric_id": 10, "operational_metric": "False Positive Hard Block Burden", "count_or_rate": f"{(three_action_res['legit_blocked']/n_test)*100:.2f}%", "percentage_of_test_cohort": f"{three_action_res['legit_blocked']} transactions"},
        {"metric_id": 11, "operational_metric": "Manual Review Workload per 1,000 Transactions", "count_or_rate": f"{(three_action_res['n_review']/n_test)*1000.0:.1f}", "percentage_of_test_cohort": f"{three_action_res['review_workload_pct']:.2f}% capacity demand"},
        {"metric_id": 12, "operational_metric": "Expected Business Cost per 1,000 Transactions", "count_or_rate": f"{three_action_res['cost_per_1k']:.2f} CU", "percentage_of_test_cohort": f"Total: {three_action_res['total_cost']:.2f} CU"},
    ]
    df_operational = pd.DataFrame(sim_test_metrics)

    # -------------------------------------------------------------
    # 6. Export Report Tables
    # -------------------------------------------------------------
    os.makedirs("reports/tables", exist_ok=True)
    df_metrics.to_csv("reports/tables/final_test_metrics.csv", index=False)
    df_policies.to_csv("reports/tables/final_policy_comparison.csv", index=False)
    df_operational.to_csv("reports/tables/final_operational_results.csv", index=False)
    print("\nSaved final evaluation tables to reports/tables/.")

    # -------------------------------------------------------------
    # 7. Render Visualizations (300 DPI)
    # -------------------------------------------------------------
    os.makedirs("reports/figures", exist_ok=True)

    # Figure 24: PR, ROC, and Calibration Curves on Final Test
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs_test)
    axes[0].plot(fpr, tpr, color="#2980B9", linewidth=2.5, label=f"Final Test ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    axes[0].set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    axes[0].set_title("ROC Curve (Final Test Cohort)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True)

    # Subplot 2: Precision-Recall Curve
    prec_c, rec_c, _ = precision_recall_curve(y_true, probs_test)
    axes[1].plot(rec_c, prec_c, color="#27AE60", linewidth=2.5, label=f"Final Test PR (AUC = {pr_auc:.4f})")
    axes[1].axhline(y_test.mean(), color="r", linestyle="--", alpha=0.6, label=f"Prevalence ({y_test.mean()*100:.1f}%)")
    axes[1].set_xlabel("Recall", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Precision", fontsize=10, fontweight="bold")
    axes[1].set_title("Precision-Recall Curve (Final Test Cohort)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    # Subplot 3: Calibration Reliability Curve
    prob_true, prob_pred = calibration_curve(y_true, probs_test, n_bins=10, strategy="uniform")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal")
    axes[2].plot(prob_pred, prob_true, marker="s", color="#8E44AD", linewidth=2, label=f"Sigmoid (Brier={brier:.4f}, ECE={ece:.4f})")
    axes[2].set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Empirical Fraud Frequency", fontsize=10, fontweight="bold")
    axes[2].set_title("Reliability Diagram (Final Test Cohort)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper left", frameon=True)

    fig.suptitle("Independent Final Test Evaluation: Discrimination & Probability Reliability\n(Untouched Holdout Cohort, n=600)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path_curves = "reports/figures/24_final_test_pr_roc_calibration.png"
    fig.savefig(path_curves, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {path_curves}")

    # Figure 25: Operational Comparison on Final Test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pol_names = df_policies["policy_name"].tolist()

    # Panel 1: Cost per 1k Txns
    axes[0].bar(pol_names, df_policies["cost_per_1k"], color=["#7F8C8D", "#E67E22", "#27AE60"], edgecolor="black")
    axes[0].set_ylabel("Cost per 1,000 Transactions (CU)", fontsize=10, fontweight="bold")
    axes[0].set_title("Expected Business Cost per 1k Txns (Final Test)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in axes[0].patches:
        val = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, val + 200, f"{val:.1f} CU", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Fraud Capture vs. Legitimate Blocked
    x_idx = np.arange(len(pol_names))
    w = 0.35
    axes[1].bar(x_idx - w/2, df_policies["fraud_capture_rate"]*100, w, label="Fraud Capture Rate (%)", color="#2980B9", edgecolor="black")
    axes[1].bar(x_idx + w/2, df_policies["legit_blocked"], w, label="Legitimate Customers Blocked", color="#C0392B", edgecolor="black")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(pol_names, fontsize=9)
    axes[1].set_title("Fraud Capture vs. Customer False Block Insult (Final Test)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")
    axes[1].legend(loc="upper left", frameon=True)

    fig.suptitle("Independent Final Test Operational Decision Benchmark", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_ops = "reports/figures/25_final_test_operational_comparison.png"
    fig.savefig(path_ops, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {path_ops}")

    # -------------------------------------------------------------
    # 8. Fresh Process Model Reload Verification
    # -------------------------------------------------------------
    print("\nValidating production model reloading in independent session...")
    reloaded_base = joblib.load("03_fraud_model_pipeline.joblib")
    reloaded_cal = joblib.load("04_calibrated_model.joblib")
    sample_raw = X_test.head(3)

    pred_base = reloaded_base.predict_proba(sample_raw)[:, 1]
    pred_cal = reloaded_cal.predict_proba(sample_raw)[:, 1]

    print(f"Reload Verification Successful:")
    print(f"  Base pipeline output:       {np.round(pred_base, 4)}")
    print(f"  Calibrated pipeline output: {np.round(pred_cal, 4)}")

    print("\nStage 12 Final Independent Test Evaluation completed successfully!")


if __name__ == "__main__":
    run_final_test_evaluation()
