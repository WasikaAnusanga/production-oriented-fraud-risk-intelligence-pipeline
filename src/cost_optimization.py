"""
Cost-Sensitive Learning and Decision Threshold Optimization Module.

Formulates asymmetric financial loss functions in Cost Units (CU) across:
1. APPROVE: Undetected fraud incurs loss proportional to transaction amount plus chargeback fees.
2. MANUAL REVIEW: Incurs human investigator review expense, customer friction burden, and residual leakage.
3. BLOCK: Legitimate transactions incur customer insult / churn penalty; fraud incurs only administrative cost.

Provides:
- Realized cost calculation on labeled policy-validation data.
- Grid search over 1-threshold (binary) and 2-threshold (APPROVE / REVIEW / BLOCK) operational policies.
- Extensive multidimensional sensitivity analysis across friction, review cost, and fraud severity.
- Publication-grade cost surface visualizations and sensitivity heatmaps.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_business_cost_config(config_path: str = "config/business_costs.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration for business cost parameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_transaction_cost(
    y_true: int,
    action: str,
    amount: float,
    scenario_cfg: Dict[str, Any],
) -> float:
    """
    Computes the realized financial cost (in Cost Units) for an individual transaction given:
    - y_true: Ground truth binary label (0 = Legitimate, 1 = Fraudulent).
    - action: Operational decision ('APPROVE', 'REVIEW', 'BLOCK').
    - amount: Transaction monetary amount (falls back to fallback_amount if NaN).
    - scenario_cfg: Cost parameter dictionary for the active scenario.
    """
    fallback_amt = scenario_cfg.get("fallback_amount", 150.0)
    amt = fallback_amt if (np.isnan(amount) or amount < 0) else amount

    c_fp_block = scenario_cfg.get("c_fp_block", 50.0)
    c_review_base = scenario_cfg.get("c_review_base", 6.0)
    c_review_friction = scenario_cfg.get("c_review_friction", 3.0)
    c_block_admin = scenario_cfg.get("c_block_admin", 1.5)
    fraud_mult = scenario_cfg.get("fraud_loss_multiplier", 1.0)
    fraud_fixed = scenario_cfg.get("fraud_loss_fixed_overhead", 10.0)
    leakage_rate = scenario_cfg.get("review_leakage_rate", 0.15)

    fraud_loss = amt * fraud_mult + fraud_fixed

    if action == "APPROVE":
        if y_true == 1:
            # False Negative: Unintercepted fraud loss
            return fraud_loss
        else:
            # True Negative: Seamless approval
            return 0.0

    elif action == "REVIEW":
        if y_true == 1:
            # Fraud routed to review: Investigator cost + residual leakage fraud loss
            return c_review_base + (leakage_rate * fraud_loss)
        else:
            # Legitimate routed to review: Investigator cost + customer friction burden
            return c_review_base + c_review_friction

    elif action == "BLOCK":
        if y_true == 1:
            # True Positive: Fraud successfully blocked (only administrative handling cost)
            return c_block_admin
        else:
            # False Positive: False declination insult / customer churn cost + admin cost
            return c_fp_block + c_block_admin

    else:
        raise ValueError(f"Unknown operational action: {action}")


def evaluate_policy_costs(
    y_true: np.ndarray,
    probs: np.ndarray,
    amounts: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    scenario_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates an operational policy defined by (review_threshold, block_threshold) on a cohort:
    - p < review_threshold: APPROVE
    - review_threshold <= p < block_threshold: MANUAL REVIEW
    - p >= block_threshold: BLOCK

    Returns comprehensive financial, operational, and confusion metrics.
    """
    n_samples = len(y_true)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    amounts = np.asarray(amounts).astype(float)

    # Determine actions
    actions = np.full(n_samples, "APPROVE", dtype=object)
    mask_review = (probs >= review_threshold) & (probs < block_threshold)
    mask_block = probs >= block_threshold
    actions[mask_review] = "REVIEW"
    actions[mask_block] = "BLOCK"

    # Compute individual realized costs
    costs = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        costs[i] = compute_transaction_cost(
            y_true=y_true[i],
            action=actions[i],
            amount=amounts[i],
            scenario_cfg=scenario_cfg,
        )

    total_cost = float(np.sum(costs))
    cost_per_1k = float((total_cost / n_samples) * 1000.0)

    # Action counts
    n_approve = int(np.sum(actions == "APPROVE"))
    n_review = int(np.sum(actions == "REVIEW"))
    n_block = int(np.sum(actions == "BLOCK"))

    # Fraud & Legitimate breakdowns
    total_fraud = int(np.sum(y_true == 1))
    total_legit = int(np.sum(y_true == 0))

    fraud_approved = int(np.sum((actions == "APPROVE") & (y_true == 1)))
    fraud_reviewed = int(np.sum((actions == "REVIEW") & (y_true == 1)))
    fraud_blocked = int(np.sum((actions == "BLOCK") & (y_true == 1)))

    legit_approved = int(np.sum((actions == "APPROVE") & (y_true == 0)))
    legit_reviewed = int(np.sum((actions == "REVIEW") & (y_true == 0)))
    legit_blocked = int(np.sum((actions == "BLOCK") & (y_true == 0)))

    # Fraud capture rate (Review + Block capture)
    fraud_captured = fraud_reviewed + fraud_blocked
    fraud_capture_rate = float(fraud_captured / total_fraud) if total_fraud > 0 else 0.0

    # Operational metrics
    review_workload_pct = float((n_review / n_samples) * 100.0)
    block_rate_pct = float((n_block / n_samples) * 100.0)

    # Binary equivalent metrics (treating REVIEW + BLOCK as positive interception flag)
    # Predicted Positive = REVIEW or BLOCK
    pred_positive = (actions != "APPROVE").astype(int)
    tp = int(np.sum((pred_positive == 1) & (y_true == 1)))
    fp = int(np.sum((pred_positive == 1) & (y_true == 0)))
    fn = int(np.sum((pred_positive == 0) & (y_true == 1)))
    tn = int(np.sum((pred_positive == 0) & (y_true == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "total_cost": total_cost,
        "cost_per_1k": cost_per_1k,
        "fraud_capture_rate": fraud_capture_rate,
        "review_workload_pct": review_workload_pct,
        "block_rate_pct": block_rate_pct,
        "fraud_approved": fraud_approved,
        "fraud_reviewed": fraud_reviewed,
        "fraud_blocked": fraud_blocked,
        "legit_approved": legit_approved,
        "legit_reviewed": legit_reviewed,
        "legit_blocked": legit_blocked,
        "n_approve": n_approve,
        "n_review": n_review,
        "n_block": n_block,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def grid_search_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    amounts: np.ndarray,
    scenario_cfg: Dict[str, Any],
    step: float = 0.02,
    mode: str = "two_threshold",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a reproducible grid search over decision thresholds:
    - mode="binary": Searches single threshold t where review_threshold = block_threshold = t
    - mode="two_threshold": Searches 0 <= review_threshold < block_threshold <= 1

    Returns DataFrame of all evaluated configurations and dictionary of the minimum-cost configuration.
    """
    records = []
    grid_vals = np.round(np.arange(0.02, 0.98 + step, step), 4)

    if mode == "binary":
        for t in grid_vals:
            res = evaluate_policy_costs(
                y_true=y_true,
                probs=probs,
                amounts=amounts,
                review_threshold=t,
                block_threshold=t,
                scenario_cfg=scenario_cfg,
            )
            records.append(res)
    elif mode == "two_threshold":
        for i, t_rev in enumerate(grid_vals[:-1]):
            for t_blk in grid_vals[i + 1:]:
                res = evaluate_policy_costs(
                    y_true=y_true,
                    probs=probs,
                    amounts=amounts,
                    review_threshold=t_rev,
                    block_threshold=t_blk,
                    scenario_cfg=scenario_cfg,
                )
                records.append(res)
    else:
        raise ValueError(f"Unknown grid search mode: {mode}")

    df_results = pd.DataFrame(records)
    best_config = df_results.loc[df_results["total_cost"].idxmin()].to_dict()
    return df_results, best_config


def run_cost_sensitivity_analysis(
    y_true: np.ndarray,
    probs: np.ndarray,
    amounts: np.ndarray,
    base_scenario_cfg: Dict[str, Any],
    sensitivity_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Performs multidimensional parameter sensitivity sweeps:
    - False positive block penalty (c_fp_block)
    - Manual review processing cost (c_review_base)
    - Review leakage rate (review_leakage_rate)
    - Fraud loss severity multiplier (fraud_loss_multiplier)
    """
    records = []

    # 1. Sweep c_fp_block
    for fp_val in sensitivity_cfg.get("c_fp_block_values", [15.0, 30.0, 50.0, 80.0, 120.0]):
        cfg = base_scenario_cfg.copy()
        cfg["c_fp_block"] = fp_val
        _, best = grid_search_thresholds(y_true, probs, amounts, cfg, step=0.04, mode="two_threshold")
        records.append({
            "parameter": "c_fp_block",
            "parameter_value": fp_val,
            "optimal_review_th": best["review_threshold"],
            "optimal_block_th": best["block_threshold"],
            "total_cost": best["total_cost"],
            "cost_per_1k": best["cost_per_1k"],
            "fraud_capture_rate": best["fraud_capture_rate"],
            "review_workload_pct": best["review_workload_pct"],
        })

    # 2. Sweep c_review_base
    for rev_val in sensitivity_cfg.get("c_review_base_values", [2.0, 4.0, 6.0, 8.0, 12.0]):
        cfg = base_scenario_cfg.copy()
        cfg["c_review_base"] = rev_val
        _, best = grid_search_thresholds(y_true, probs, amounts, cfg, step=0.04, mode="two_threshold")
        records.append({
            "parameter": "c_review_base",
            "parameter_value": rev_val,
            "optimal_review_th": best["review_threshold"],
            "optimal_block_th": best["block_threshold"],
            "total_cost": best["total_cost"],
            "cost_per_1k": best["cost_per_1k"],
            "fraud_capture_rate": best["fraud_capture_rate"],
            "review_workload_pct": best["review_workload_pct"],
        })

    # 3. Sweep review_leakage_rate
    for leak_val in sensitivity_cfg.get("review_leakage_rates", [0.05, 0.10, 0.15, 0.25, 0.35]):
        cfg = base_scenario_cfg.copy()
        cfg["review_leakage_rate"] = leak_val
        _, best = grid_search_thresholds(y_true, probs, amounts, cfg, step=0.04, mode="two_threshold")
        records.append({
            "parameter": "review_leakage_rate",
            "parameter_value": leak_val,
            "optimal_review_th": best["review_threshold"],
            "optimal_block_th": best["block_threshold"],
            "total_cost": best["total_cost"],
            "cost_per_1k": best["cost_per_1k"],
            "fraud_capture_rate": best["fraud_capture_rate"],
            "review_workload_pct": best["review_workload_pct"],
        })

    # 4. Sweep fraud_loss_multiplier
    for mult_val in sensitivity_cfg.get("fraud_multiplier_values", [0.70, 0.85, 1.0, 1.25, 1.50]):
        cfg = base_scenario_cfg.copy()
        cfg["fraud_loss_multiplier"] = mult_val
        _, best = grid_search_thresholds(y_true, probs, amounts, cfg, step=0.04, mode="two_threshold")
        records.append({
            "parameter": "fraud_loss_multiplier",
            "parameter_value": mult_val,
            "optimal_review_th": best["review_threshold"],
            "optimal_block_th": best["block_threshold"],
            "total_cost": best["total_cost"],
            "cost_per_1k": best["cost_per_1k"],
            "fraud_capture_rate": best["fraud_capture_rate"],
            "review_workload_pct": best["review_workload_pct"],
        })

    return pd.DataFrame(records)


def plot_threshold_cost_surfaces(
    binary_grid_df: pd.DataFrame,
    two_th_grid_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> Dict[str, str]:
    """
    Renders publication-quality visualizations for cost-sensitive optimization:
    1. 1D Binary Threshold vs. Total Cost Curve.
    2. 2D Two-Threshold Operational Cost Surface Heatmap.
    3. Trade-off Curve: Fraud Capture Rate vs. Manual Review Workload.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}

    # Figure 1: 1D Binary Threshold vs Total Cost
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2C3E50"
    color2 = "#E74C3C"

    best_binary = binary_grid_df.loc[binary_grid_df["total_cost"].idxmin()]

    ax1.plot(
        binary_grid_df["block_threshold"],
        binary_grid_df["cost_per_1k"],
        color=color1,
        linewidth=2.5,
        label="Cost per 1k Txns (CU)",
    )
    ax1.scatter(
        [best_binary["block_threshold"]],
        [best_binary["cost_per_1k"]],
        color=color2,
        s=100,
        zorder=5,
        label=f"Optimum: t={best_binary['block_threshold']:.2f} ({best_binary['cost_per_1k']:.1f} CU)",
    )
    ax1.axvline(0.50, color="#7F8C8D", linestyle="--", alpha=0.8, label="Default t=0.50")

    ax1.set_xlabel("Binary Decision Threshold (t)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cost per 1,000 Transactions (Cost Units)", fontsize=11, fontweight="bold", color=color1)
    ax1.set_title("Binary Decision Threshold vs. Financial Cost Surface\n(Policy-Validation Cohort, n=600)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    path_binary = os.path.join(output_dir, "09_binary_threshold_cost_curve.png")
    fig.savefig(path_binary, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["binary_curve"] = path_binary

    # Figure 2: 2D Two-Threshold Operational Cost Surface Heatmap
    pivot_costs = two_th_grid_df.pivot(
        index="review_threshold", columns="block_threshold", values="cost_per_1k"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_costs,
        cmap="viridis_r",
        annot=False,
        cbar_kws={"label": "Cost per 1,000 Txns (CU)"},
        ax=ax,
    )
    best_2th = two_th_grid_df.loc[two_th_grid_df["total_cost"].idxmin()]
    ax.set_title(
        f"Two-Threshold Operational Policy Cost Surface\nOptimal: Review={best_2th['review_threshold']:.2f}, Block={best_2th['block_threshold']:.2f} (Cost: {best_2th['cost_per_1k']:.1f} CU/1k)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Block Threshold ($t_{block}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Review Threshold ($t_{review}$)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    path_heatmap = os.path.join(output_dir, "10_two_threshold_cost_surface_heatmap.png")
    fig.savefig(path_heatmap, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["two_th_heatmap"] = path_heatmap

    # Figure 3: Fraud Capture vs Review Workload Trade-off
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        two_th_grid_df["review_workload_pct"],
        two_th_grid_df["fraud_capture_rate"] * 100.0,
        c=two_th_grid_df["cost_per_1k"],
        cmap="coolwarm_r",
        alpha=0.7,
        edgecolors="none",
        s=30,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cost per 1,000 Txns (CU)", fontsize=10, fontweight="bold")

    # Highlight optimal point
    ax.scatter(
        [best_2th["review_workload_pct"]],
        [best_2th["fraud_capture_rate"] * 100.0],
        color="#F1C40F",
        edgecolor="black",
        s=140,
        label=f"Cost-Optimal Policy ({best_2th['fraud_capture_rate']*100:.1f}% Capture, {best_2th['review_workload_pct']:.1f}% Review)",
        zorder=10,
    )

    ax.set_xlabel("Manual Review Workload (% of All Transactions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraud Capture Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Operational Tradeoff: Fraud Capture vs. Manual Review Capacity", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    path_tradeoff = os.path.join(output_dir, "11_fraud_capture_vs_review_workload.png")
    fig.savefig(path_tradeoff, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["tradeoff"] = path_tradeoff

    return generated_figures


def plot_sensitivity_heatmaps(
    sensitivity_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """Renders parameter sensitivity summary charts."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    params = [
        ("c_fp_block", "False Positive Block Penalty (CU)"),
        ("c_review_base", "Manual Review Base Cost (CU)"),
        ("review_leakage_rate", "Review Leakage Rate"),
        ("fraud_loss_multiplier", "Fraud Loss Multiplier"),
    ]

    for idx, (param, title) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        sub = sensitivity_df[sensitivity_df["parameter"] == param]

        ax.plot(
            sub["parameter_value"],
            sub["cost_per_1k"],
            marker="o",
            color="#2980B9",
            linewidth=2,
            label="Cost per 1k Txns",
        )
        ax.set_xlabel(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Cost per 1k Txns (CU)", fontsize=10, fontweight="bold", color="#2980B9")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Twin axis for review threshold
        ax2 = ax.twinx()
        ax2.plot(
            sub["parameter_value"],
            sub["optimal_review_th"],
            marker="s",
            color="#E67E22",
            linestyle="--",
            label="Optimal $t_{review}$",
        )
        ax2.set_ylabel("Optimal $t_{review}$", fontsize=10, fontweight="bold", color="#E67E22")

        ax.set_title(f"Sensitivity to {param}", fontsize=11, fontweight="bold")

    fig.suptitle("Multidimensional Cost Sensitivity Analysis\n(Impact of Operating Parameters on Thresholds and Costs)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_sens = os.path.join(output_dir, "12_cost_sensitivity_multipanel.png")
    fig.savefig(path_sens, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_sens


def compute_financial_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_fn: float = 100.0,
    c_fp: float = 50.0,
) -> Dict[str, float]:
    """
    Standard binary cost-matrix evaluation utility for backward compatibility:
    Total Loss = FN * c_fn + FP * c_fp
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    total_loss = float(fn * c_fn + fp * c_fp)
    return {
        "total_financial_loss": total_loss,
        "fn_loss": float(fn * c_fn),
        "fp_loss": float(fp * c_fp),
        "false_negatives": fn,
        "false_positives": fp,
    }


