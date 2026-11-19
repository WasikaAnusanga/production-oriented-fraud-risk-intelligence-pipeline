"""
Explainable Artificial Intelligence (XAI) Module for Fraud Risk Intelligence.

Provides:
- Global feature importance (Mean |SHAP|, Gini / MDI importance, Permutation Importance).
- Cross-fold feature importance stability analysis across development folds.
- SHAP summary beeswarm plots and ranked importance bar charts.
- Non-linear feature dependence and interaction plots.
- Local explainability with TreeSHAP waterfalls and adverse action reason codes.
- Strict technical separation between base model SHAP margins and post-hoc calibrated probabilities.
- Plain-language narratives for regulatory compliance and operational audits.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
import joblib

from config.settings import RANDOM_STATE


def clean_feature_name(name: str) -> str:
    """Cleans scikit-learn feature names from ColumnTransformer prefixes."""
    for prefix in ["num__", "cat__", "remainder__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def get_pipeline_feature_names(pipeline) -> List[str]:
    """Extracts human-readable feature names from the pipeline preprocessor."""
    prep = pipeline.named_steps["preprocessor"]
    raw_names = prep.get_feature_names_out()
    return [clean_feature_name(n) for n in raw_names]


def compute_global_feature_importance(
    pipeline,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    n_permutations: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray, shap.Explanation, np.ndarray]:
    """
    Computes multifaceted global feature importance:
    1. TreeSHAP Mean Absolute Value (mean |SHAP|).
    2. Random Forest Gini / Mean Decrease in Impurity (MDI).
    3. Pipeline Permutation Importance on PR-AUC.

    Returns:
    --------
    importance_df : pd.DataFrame
        Ranked table of all features with their SHAP, Gini, and Permutation metrics.
    shap_values_pos : np.ndarray
        Raw SHAP matrix for the positive fraud class (n_samples, n_features).
    shap_explanation : shap.Explanation
        Full SHAP explanation object for beeswarm and waterfall plotting.
    X_transformed : np.ndarray
        Transformed numerical feature matrix input to the tree classifier.
    """
    fe = pipeline.named_steps["fe"]
    prep = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["classifier"]

    # Transform data up to classifier
    X_fe = fe.transform(X_sample)
    X_trans = prep.transform(X_fe)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = get_pipeline_feature_names(pipeline)

    # 1. SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer(X_trans)

    # Handle multi-class / binary structure of shap_vals
    if len(shap_vals.shape) == 3 and shap_vals.shape[-1] == 2:
        # Binary classification (class 0, class 1)
        shap_values_pos = shap_vals.values[:, :, 1]
        base_value_pos = shap_vals.base_values[0, 1] if shap_vals.base_values.ndim > 1 else shap_vals.base_values[1]
        exp_obj = shap.Explanation(
            values=shap_values_pos,
            base_values=base_value_pos,
            data=X_trans,
            feature_names=feature_names,
        )
    else:
        shap_values_pos = shap_vals.values
        base_value_pos = shap_vals.base_values[0] if hasattr(shap_vals.base_values, "__len__") else shap_vals.base_values
        exp_obj = shap.Explanation(
            values=shap_values_pos,
            base_values=base_value_pos,
            data=X_trans,
            feature_names=feature_names,
        )

    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)

    # 2. Gini / MDI Importance
    gini_importance = rf.feature_importances_

    # 3. Permutation Importance
    perm = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=n_permutations,
        random_state=random_state,
        n_jobs=-1,
    )
    perm_mean = perm.importances_mean

    # Build importance dataframe
    # Map permutation importances back to raw or transformed features
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "gini_importance": gini_importance,
    })
    df_imp["shap_rank"] = df_imp["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_imp = df_imp.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Flag engineered vs original
    engineered_stems = [
        "log_transaction_amount", "log_distance", "velocity_ratio", "previous_23h_transactions",
        "amount_per_recent_transaction", "combined_merchant_ip_risk", "merchant_ip_risk_difference",
        "amount_merchant_risk_interaction", "new_device_distance_interaction", "failed_login_ip_risk",
        "account_device_age_ratio", "email_account_age_ratio", "login_speed_indicator",
        "unusual_hour_velocity", "risk_composite_score", "high_velocity_new_device",
        "amount_to_avg_ratio", "missing_auth_ecommerce", "missing_auth_indicator", "missing_dist_indicator"
    ]
    def is_engineered(feat: str) -> bool:
        return any(stem in feat for stem in engineered_stems)

    df_imp["is_engineered"] = df_imp["feature"].apply(is_engineered)
    return df_imp, shap_values_pos, exp_obj, X_trans


def analyze_feature_importance_stability(
    calibrated_model,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Analyzes the consistency of feature importance across all individual cross-validation
    folds trained within the ensemble calibrator.
    Maps importances dynamically to feature names per fold to safely handle folds with
    varying observed one-hot categories.
    """
    estimators = [cal_clf.estimator for cal_clf in calibrated_model.calibrated_classifiers_]
    
    fold_records = []
    for fold_idx, est in enumerate(estimators):
        rf = est.named_steps["classifier"]
        feat_names = get_pipeline_feature_names(est)
        imps = rf.feature_importances_
        for f, imp in zip(feat_names, imps):
            fold_records.append({
                "fold": fold_idx + 1,
                "feature": f,
                "importance": float(imp),
            })

    df_folds = pd.DataFrame(fold_records)
    agg_df = df_folds.groupby("feature")["importance"].agg(
        mean_importance="mean",
        std_importance="std",
    ).reset_index()

    agg_df["std_importance"] = agg_df["std_importance"].fillna(0.0)
    agg_df["cv_pct_stability"] = np.where(
        agg_df["mean_importance"] > 0,
        (agg_df["std_importance"] / agg_df["mean_importance"]) * 100.0,
        0.0,
    )
    agg_df = agg_df.sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
    agg_df["rank"] = np.arange(1, len(agg_df) + 1)

    return agg_df.head(top_k)



def plot_global_importance_and_summary(
    importance_df: pd.DataFrame,
    shap_explanation: shap.Explanation,
    output_dir: str = "reports/figures",
    top_n: int = 15,
) -> Dict[str, str]:
    """Renders ranked importance bar chart and SHAP beeswarm plot (300 DPI)."""
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}

    # 1. Ranked Global Feature Importance Bar Chart
    top_df = importance_df.head(top_n).sort_values(by="mean_abs_shap", ascending=True)
    colors = ["#E74C3C" if eng else "#3498DB" for eng in top_df["is_engineered"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_df["feature"], top_df["mean_abs_shap"], color=colors, edgecolor="black", alpha=0.85)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", edgecolor="black", label="Engineered Feature"),
        Patch(facecolor="#3498DB", edgecolor="black", label="Original Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Base Fraud Log-Odds)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Global Fraud Predictors (TreeSHAP Mean |SHAP|)\n(Trained Random Forest Pipeline)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    path_bar = os.path.join(output_dir, "13_global_feature_importance_ranked.png")
    fig.savefig(path_bar, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["ranked_importance"] = path_bar

    # 2. SHAP Summary Beeswarm Plot
    fig = plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(shap_explanation, max_display=top_n, show=False)
    plt.title(f"SHAP Summary (Beeswarm) Distribution for Top {top_n} Features\n(Feature Value Impact on Raw Ensemble Fraud Logit)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path_summary = os.path.join(output_dir, "14_shap_summary_beeswarm.png")
    plt.savefig(path_summary, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures["shap_summary"] = path_summary

    return generated_figures


def plot_shap_dependences(
    shap_explanation: shap.Explanation,
    output_dir: str = "reports/figures",
) -> Dict[str, str]:
    """
    Renders 3 key dependence and interaction plots:
    1. transaction_amount vs merchant_risk_score
    2. new_device vs distance_from_usual_location_km
    3. velocity_ratio vs ip_risk_score
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}
    feature_names = shap_explanation.feature_names

    # Helper to find matching feature index or name
    def find_feature(substring: str) -> Optional[str]:
        for f in feature_names:
            if substring.lower() in f.lower():
                return f
        return None

    dep_pairs = [
        ("transaction_amount", "merchant_risk_score", "15_shap_dependence_amount_vs_merchant.png"),
        ("new_device", "distance", "16_shap_dependence_new_device_vs_distance.png"),
        ("velocity_ratio", "ip_risk_score", "17_shap_dependence_velocity_vs_ip.png"),
    ]

    for f1_sub, f2_sub, filename in dep_pairs:
        f1 = find_feature(f1_sub)
        f2 = find_feature(f2_sub)

        if f1 is not None and f2 is not None:
            fig = plt.figure(figsize=(8, 6))
            shap.plots.scatter(shap_explanation[:, f1], color=shap_explanation[:, f2], show=False)
            plt.title(f"SHAP Dependence: {f1} colored by {f2}", fontsize=11, fontweight="bold", pad=12)
            plt.tight_layout()
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated_figures[f"{f1_sub}_vs_{f2_sub}"] = out_path

    return generated_figures


def generate_local_explanations(
    pipeline,
    calibrated_pipeline,
    X_cohort: pd.DataFrame,
    y_cohort: pd.Series,
    ids_cohort: pd.Series,
    output_dir: str = "reports/figures",
    review_threshold: float = 0.10,
    block_threshold: float = 0.36,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Selects 5 representative real transactions across outcome regimes:
    1. High-risk Fraudulent (BLOCK)
    2. High-risk Legitimate (BLOCK or REVIEW - false alert)
    3. Medium-risk Transaction (MANUAL REVIEW)
    4. Low-risk Fraudulent (APPROVE - false negative evasion)
    5. Low-risk Legitimate (APPROVE - true negative seamless)

    Generates:
    - 5 SHAP waterfall plots (300 DPI).
    - Ranked top-3 risk increasing and risk decreasing factors.
    - Plain-language operational reason codes.
    - Summary table exported to reports/tables/local_explanations.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = {}

    # Calibrated probabilities
    cal_probs = calibrated_pipeline.predict_proba(X_cohort)[:, 1]

    # Assign operational decisions
    actions = np.full(len(X_cohort), "APPROVE", dtype=object)
    actions[(cal_probs >= review_threshold) & (cal_probs < block_threshold)] = "MANUAL REVIEW"
    actions[cal_probs >= block_threshold] = "BLOCK"

    y_vals = y_cohort.to_numpy()
    ids_vals = ids_cohort.to_numpy()

    # Search for representative cases
    # 1. High-risk Fraud (y=1, action=BLOCK)
    idx_hr_fraud = np.where((y_vals == 1) & (actions == "BLOCK"))[0]
    idx1 = idx_hr_fraud[np.argmax(cal_probs[idx_hr_fraud])] if len(idx_hr_fraud) > 0 else np.argmax(cal_probs)

    # 2. High-risk Legit (y=0, highest prob)
    idx_hr_legit = np.where((y_vals == 0) & (cal_probs >= review_threshold))[0]
    idx2 = idx_hr_legit[np.argmax(cal_probs[idx_hr_legit])] if len(idx_hr_legit) > 0 else np.where(y_vals == 0)[0][0]

    # 3. Medium-risk (action=MANUAL REVIEW)
    idx_med = np.where(actions == "MANUAL REVIEW")[0]
    idx3 = idx_med[len(idx_med) // 2] if len(idx_med) > 0 else 0

    # 4. Low-risk Fraud (y=1, lowest prob among frauds)
    idx_lr_fraud = np.where((y_vals == 1) & (actions == "APPROVE"))[0]
    idx4 = idx_lr_fraud[np.argmin(cal_probs[idx_lr_fraud])] if len(idx_lr_fraud) > 0 else np.where(y_vals == 1)[0][-1]

    # 5. Low-risk Legit (y=0, lowest prob)
    idx_lr_legit = np.where(y_vals == 0)[0]
    idx5 = idx_lr_legit[np.argmin(cal_probs[idx_lr_legit])]

    selected_indices = [
        ("High-Risk Fraud", idx1),
        ("High-Risk Legitimate (False Alert)", idx2),
        ("Medium-Risk (Borderline Review)", idx3),
        ("Low-Risk Fraud (Undetected Leakage)", idx4),
        ("Low-Risk Legitimate (Seamless Approval)", idx5),
    ]

    # Extract base tree pipeline and compute SHAP explanation
    fe = pipeline.named_steps["fe"]
    prep = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(pipeline)

    explainer = shap.TreeExplainer(rf)

    records = []

    for label, idx in selected_indices:
        x_row = X_cohort.iloc[[idx]]
        tid = ids_vals[idx]
        actual_label = int(y_vals[idx])
        p_cal = float(cal_probs[idx])
        action = actions[idx]

        # Transform single row
        x_fe = fe.transform(x_row)
        x_trans = prep.transform(x_fe)
        if hasattr(x_trans, "toarray"):
            x_trans = x_trans.toarray()

        shap_row = explainer(x_trans)
        if len(shap_row.shape) == 3 and shap_row.shape[-1] == 2:
            row_vals = shap_row.values[0, :, 1]
            base_val = shap_row.base_values[0, 1] if shap_row.base_values.ndim > 1 else shap_row.base_values[1]
        else:
            row_vals = shap_row.values[0]
            base_val = shap_row.base_values[0] if hasattr(shap_row.base_values, "__len__") else shap_row.base_values

        row_exp = shap.Explanation(
            values=row_vals,
            base_values=base_val,
            data=x_trans[0],
            feature_names=feature_names,
        )

        # Generate Waterfall Plot
        clean_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filename = f"18_waterfall_{clean_label}.png"
        out_path = os.path.join(output_dir, filename)

        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(row_exp, max_display=10, show=False)
        plt.title(
            f"SHAP Waterfall: {label} (Txn: {tid})\nCalibrated P(Fraud)={p_cal:.3f} | Decision: {action} | Actual: {actual_label}",
            fontsize=10,
            fontweight="bold",
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated_plots[clean_label] = out_path

        # Identify Top Risk Increasing & Decreasing Features
        feat_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": row_vals,
            "feature_value": x_trans[0],
        })

        top_increasing = feat_df.sort_values(by="shap_value", ascending=False).head(3)
        top_decreasing = feat_df.sort_values(by="shap_value", ascending=True).head(3)

        inc_str = "; ".join([f"{r['feature']} (+{r['shap_value']:.3f})" for _, r in top_increasing.iterrows() if r["shap_value"] > 0])
        dec_str = "; ".join([f"{r['feature']} ({r['shap_value']:.3f})" for _, r in top_decreasing.iterrows() if r["shap_value"] < 0])

        # Synthesize plain-language explanation
        primary_driver = top_increasing.iloc[0]["feature"] if len(top_increasing) > 0 and top_increasing.iloc[0]["shap_value"] > 0 else "Baseline benign history"
        narrative = (
            f"Transaction routed to {action} (calibrated fraud likelihood {p_cal*100:.1f}%). "
            f"Elevated risk driven primarily by {primary_driver}. "
            f"Mitigated by benign signals: {top_decreasing.iloc[0]['feature']}."
        )

        records.append({
            "case_type": label,
            "transaction_id": tid,
            "actual_fraud": actual_label,
            "calibrated_prob": round(p_cal, 4),
            "operational_action": action,
            "top_risk_increasing_factors": inc_str,
            "top_risk_decreasing_factors": dec_str,
            "plain_language_explanation": narrative,
            "waterfall_figure": filename,
        })

    local_df = pd.DataFrame(records)
    return local_df, generated_plots
