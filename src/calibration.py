"""
Probability Calibration and Reliability Engineering Module for Fraud Risk Intelligence.

This module provides statistically sound calibration methods (Platt / Sigmoid Scaling and Isotonic Regression)
with strict cross-validation safeguards to prevent data leakage and optimism bias:
- CalibratedClassifierCV wrapping full end-to-end pipelines (supporting raw input DataFrames).
- Out-of-fold calibration evaluation across stratified folds.
- Calculation of Brier score, Log Loss, Expected Calibration Error (ECE), ROC-AUC, and Average Precision.
- Publication-quality reliability diagrams and probability density comparisons.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config.settings import RANDOM_STATE, RAW_DATA_PATH
from src.data_splitting import get_partition_data


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Expected Calibration Error (ECE) and bin statistics:
    
    ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|
    
    where:
    - B_m is the set of samples whose predicted probabilities fall in bin m.
    - |B_m| is the number of samples in bin m.
    - N is the total number of samples.
    - acc(B_m) is the empirical accuracy (fraction of positives) in bin m.
    - conf(B_m) is the mean predicted probability (confidence) in bin m.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted positive-class probabilities in [0, 1].
    n_bins : int, default=10
        Number of bins for partitioning predicted probabilities.
    strategy : str, default="uniform"
        Binning strategy: 'uniform' (equal width [0, 1]) or 'quantile' (equal frequency).

    Returns:
    --------
    ece : float
        Scalar Expected Calibration Error in [0, 1].
    prob_true : ndarray
        Empirical positive rate in each populated bin.
    prob_pred : ndarray
        Mean predicted probability in each populated bin.
    bin_counts : ndarray
        Number of samples assigned to each populated bin.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[-1] += 1e-8
        bin_ids = np.digitize(y_prob, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    prob_true_list = []
    prob_pred_list = []
    bin_counts_list = []
    total_samples = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = (bin_ids == b)
        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            prob_true_list.append(bin_acc)
            prob_pred_list.append(bin_conf)
            bin_counts_list.append(count)
            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece), np.array(prob_true_list), np.array(prob_pred_list), np.array(bin_counts_list)


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Convenience wrapper returning scalar ECE value directly.
    """
    ece, _, _, _ = compute_expected_calibration_error(
        y_true=y_true, y_prob=y_prob, n_bins=n_bins, strategy=strategy
    )
    return float(ece)



def evaluate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes a comprehensive suite of probability calibration and discrimination metrics:
    - Brier score
    - Log loss (cross-entropy)
    - Expected Calibration Error (ECE)
    - ROC-AUC
    - Average Precision (PR-AUC)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)

    brier = brier_score_loss(y_true, y_prob)
    loss = log_loss(y_true, y_prob_clipped)
    ece, _, _, _ = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "brier_score": float(brier),
        "log_loss": float(loss),
        "ece": float(ece),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }


def cross_validate_calibration(
    base_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Performs rigorous out-of-fold cross-validation across development data to compare:
    1. Uncalibrated predictions
    2. Sigmoid (Platt) calibrated predictions
    3. Isotonic calibrated predictions

    Safeguards:
    - For each CV split (outer fold), a model is trained on outer train.
    - Inside outer train, CalibratedClassifierCV(cv=3) or nested cross-fitting is used
      to fit calibration curves so that calibration parameters are NOT fitted on the exact
      same samples without separation.
    - Predictions are collected out-of-fold on the outer test split.
    - Per-fold variability (mean ± std) is computed.

    Returns:
    --------
    summary_df : pd.DataFrame
        Aggregate mean ± std for each metric across folds.
    oof_predictions : Dict[str, np.ndarray]
        Concatenated out-of-fold predicted probabilities for each method.
    fold_df : pd.DataFrame
        Detailed metric values for every individual fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    methods = ["Uncalibrated", "Sigmoid", "Isotonic"]
    oof_preds = {m: np.zeros(len(y), dtype=float) for m in methods}
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 1. Uncalibrated: Train base pipeline on train fold
        uncalibrated_pipe = clone(base_pipeline)
        uncalibrated_pipe.fit(X_train, y_train)
        p_uncal = uncalibrated_pipe.predict_proba(X_val)[:, 1]
        oof_preds["Uncalibrated"][val_idx] = p_uncal

        # 2. Sigmoid Calibration using 3-fold inner cross-fitting inside X_train
        sigmoid_clf = CalibratedClassifierCV(
            estimator=clone(base_pipeline),
            method="sigmoid",
            cv=3,
        )
        sigmoid_clf.fit(X_train, y_train)
        p_sig = sigmoid_clf.predict_proba(X_val)[:, 1]
        oof_preds["Sigmoid"][val_idx] = p_sig

        # 3. Isotonic Calibration using 3-fold inner cross-fitting inside X_train
        isotonic_clf = CalibratedClassifierCV(
            estimator=clone(base_pipeline),
            method="isotonic",
            cv=3,
        )
        isotonic_clf.fit(X_train, y_train)
        p_iso = isotonic_clf.predict_proba(X_val)[:, 1]
        oof_preds["Isotonic"][val_idx] = p_iso

        # Compute fold metrics
        for method, p_val in [("Uncalibrated", p_uncal), ("Sigmoid", p_sig), ("Isotonic", p_iso)]:
            metrics = evaluate_calibration_metrics(y_val.to_numpy(), p_val)
            fold_records.append({
                "fold": fold_idx + 1,
                "method": method,
                **metrics,
            })

    fold_df = pd.DataFrame(fold_records)

    # Compute summary statistics (mean and std across folds)
    summary_list = []
    for method in methods:
        m_df = fold_df[fold_df["method"] == method]
        row = {
            "Method": method,
            "Brier_mean": m_df["brier_score"].mean(),
            "Brier_std": m_df["brier_score"].std(),
            "LogLoss_mean": m_df["log_loss"].mean(),
            "LogLoss_std": m_df["log_loss"].std(),
            "ECE_mean": m_df["ece"].mean(),
            "ECE_std": m_df["ece"].std(),
            "ROC_AUC_mean": m_df["roc_auc"].mean(),
            "ROC_AUC_std": m_df["roc_auc"].std(),
            "PR_AUC_mean": m_df["pr_auc"].mean(),
            "PR_AUC_std": m_df["pr_auc"].std(),
        }
        summary_list.append(row)

    summary_df = pd.DataFrame(summary_list)
    return summary_df, oof_preds, fold_df


def fit_final_calibrated_pipeline(
    base_pipeline,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Fits a production CalibratedClassifierCV object across all development data using internal CV.
    
    The resulting calibrated classifier:
    - Wraps the full preprocessing and feature engineering steps inside base_pipeline.
    - Accepts raw transaction DataFrames as input during inference.
    - Employs internal cross-fitting (e.g. 5-fold ensemble of calibrators) on development data.
    """
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(base_pipeline),
        method=method,
        cv=cv,
    )
    calibrated_model.fit(X_dev, y_dev)
    return calibrated_model


def plot_reliability_diagrams(
    y_true: np.ndarray,
    oof_predictions: Dict[str, np.ndarray],
    output_dir: str = "reports/figures",
    n_bins: int = 10,
) -> Dict[str, str]:
    """
    Generates publication-quality calibration plots (300 DPI):
    1. Uncalibrated reliability diagram
    2. Sigmoid reliability diagram
    3. Isotonic reliability diagram
    4. Combined calibration curves
    5. Predicted probability distributions for legitimate vs fraudulent transactions
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}

    palette = {
        "Uncalibrated": "#3498DB",  # Blue
        "Sigmoid": "#2ECC71",       # Green
        "Isotonic": "#E74C3C",      # Red
    }

    # Individual reliability diagrams (1 to 3)
    for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
        p = oof_predictions[method]
        ece, prob_true, prob_pred, bin_counts = compute_expected_calibration_error(
            y_true, p, n_bins=n_bins
        )
        brier = brier_score_loss(y_true, p)
        logloss = log_loss(y_true, np.clip(p, 1e-15, 1 - 1e-15))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Reliability curve
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        ax1.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            color=palette[method],
            label=f"{method} (ECE={ece:.4f}, Brier={brier:.4f})",
        )
        ax1.set_ylabel("Empirical Fraud Frequency", fontsize=11, fontweight="bold")
        ax1.set_title(f"Reliability Diagram: {method} Classifier\n(Development Out-of-Fold, n=1,800)", fontsize=12, fontweight="bold", pad=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)
        ax1.set_ylim(-0.02, 1.02)

        # Histogram of predictions
        ax2.hist(p, bins=np.linspace(0, 1, n_bins + 1), color=palette[method], alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Predicted Fraud Probability", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_yscale("log")
        ax2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        filename = f"calibration_reliability_{method.lower()}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated_figures[f"reliability_{method.lower()}"] = path

    # 4. Combined Calibration Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7, linewidth=1.5)

    for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
        p = oof_predictions[method]
        ece, prob_true, prob_pred, _ = compute_expected_calibration_error(y_true, p, n_bins=n_bins)
        brier = brier_score_loss(y_true, p)
        ax.plot(
            prob_pred,
            prob_true,
            marker="s" if method == "Sigmoid" else ("^" if method == "Isotonic" else "o"),
            linewidth=2,
            label=f"{method} (ECE={ece:.4f}, Brier={brier:.4f})",
            color=palette[method],
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Fraud Fraction", fontsize=11, fontweight="bold")
    ax.set_title("Calibration Curve Comparison across Scaling Methods\n(Development Out-of-Fold 5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    path_combined = os.path.join(output_dir, "07_calibration_comparison_curves.png")
    fig.savefig(path_combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["combined_curves"] = path_combined

    # 5. Predicted probability distributions for legitimate vs fraudulent transactions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    methods = ["Uncalibrated", "Sigmoid", "Isotonic"]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        p = oof_predictions[method]
        legit_probs = p[y_true == 0]
        fraud_probs = p[y_true == 1]

        sns.kdeplot(legit_probs, ax=ax, label=f"Legitimate (n={len(legit_probs)})", color="#27AE60", fill=True, alpha=0.4, clip=(0, 1))
        sns.kdeplot(fraud_probs, ax=ax, label=f"Fraudulent (n={len(fraud_probs)})", color="#C0392B", fill=True, alpha=0.4, clip=(0, 1))

        ax.set_title(f"{method} Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Fraud Probability", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.set_xlim(0, 1)

    fig.suptitle("Predicted Fraud Probability Densities: Legitimate vs. Fraudulent Transactions\n(5-Fold Out-of-Fold Development Partition)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path_dist = os.path.join(output_dir, "08_calibrated_probability_distributions.png")
    fig.savefig(path_dist, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["probability_distributions"] = path_dist

    return generated_figures
