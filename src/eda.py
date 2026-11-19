"""
Exploratory Data Analysis (EDA) Module.

Provides comprehensive functions for univariate, bivariate, temporal,
non-linear, and multi-variable interaction analyses.
Generates statistical tables and publication-quality figures (300 DPI)
strictly on the Development Partition.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import FIGURES_DIR, TABLES_DIR, TARGET_COL

# Apply consistent aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def compute_categorical_risk_profiles(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = TARGET_COL,
    fill_missing: Optional[str] = "MISSING"
) -> pd.DataFrame:
    """
    Computes transaction volume, fraud count, empirical fraud rate,
    standard error of proportion, and relative risk ratio across categorical levels.
    """
    s = df[cat_col].copy()
    if fill_missing is not None:
        s = s.fillna(fill_missing).astype(str)
    else:
        s = s.astype(str)

    temp_df = pd.DataFrame({cat_col: s, target_col: df[target_col]})
    base_rate = float(df[target_col].mean())

    summary = temp_df.groupby(cat_col)[target_col].agg(
        volume="count",
        fraud_count="sum"
    ).reset_index()

    summary["fraud_rate"] = summary["fraud_count"] / summary["volume"]
    summary["fraud_rate_pct"] = (summary["fraud_rate"] * 100).round(2)
    summary["volume_pct"] = ((summary["volume"] / len(df)) * 100).round(2)
    summary["relative_risk"] = (summary["fraud_rate"] / base_rate).round(2) if base_rate > 0 else 1.0

    # Normal approximation standard error for proportions
    summary["se_pct"] = (
        np.sqrt(summary["fraud_rate"] * (1.0 - summary["fraud_rate"]) / summary["volume"]) * 100
    ).round(2)

    return summary.sort_values(by="fraud_rate", ascending=False).reset_index(drop=True)


def compute_binned_numerical_risk_profiles(
    df: pd.DataFrame,
    num_col: str,
    n_bins: int = 5,
    qcut: bool = True,
    target_col: str = TARGET_COL,
    fill_missing_val: Optional[float] = None
) -> pd.DataFrame:
    """
    Bins a numeric column into quantile or equal-width bins on the development set
    and computes volume, fraud count, fraud rate, and relative risk ratio.
    """
    s = df[num_col].copy()
    if fill_missing_val is not None:
        s = s.fillna(fill_missing_val)
    valid_mask = s.notnull()
    s_valid = s[valid_mask]
    y_valid = df.loc[valid_mask, target_col]

    base_rate = float(y_valid.mean())

    if qcut:
        try:
            bins = pd.qcut(s_valid, q=n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(s_valid, bins=n_bins)
    else:
        bins = pd.cut(s_valid, bins=n_bins)

    temp_df = pd.DataFrame({"bin": bins.astype(str), target_col: y_valid})
    summary = temp_df.groupby("bin")[target_col].agg(
        volume="count",
        fraud_count="sum"
    ).reset_index()

    summary["fraud_rate"] = summary["fraud_count"] / summary["volume"]
    summary["fraud_rate_pct"] = (summary["fraud_rate"] * 100).round(2)
    summary["volume_pct"] = ((summary["volume"] / len(s_valid)) * 100).round(2)
    summary["relative_risk"] = (summary["fraud_rate"] / base_rate).round(2) if base_rate > 0 else 1.0

    return summary


def plot_dual_axis_rate_and_volume(
    summary_df: pd.DataFrame,
    category_col: str,
    title: str,
    filename: str,
    base_rate: float = 12.11,
    fig_dir: Path = FIGURES_DIR,
    rot: int = 0
) -> Path:
    """
    Standardizes a publication-quality dual-axis chart displaying transaction volume (bars)
    alongside empirical fraud rate (line/markers) with base rate reference line.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2 = ax1.twinx()

    x = np.arange(len(summary_df))
    bars = ax1.bar(x, summary_df["volume"], color="#90afc7", width=0.55, alpha=0.85, label="Transaction Volume")
    line = ax2.plot(x, summary_df["fraud_rate_pct"], color="#b23a22", marker="o", linewidth=2.5, markersize=7, label="Fraud Rate (%)")

    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", linewidth=1.5, alpha=0.9, label=f"Dev Base Rate ({base_rate:.2f}%)")

    # Annotate rates
    for i, rate in enumerate(summary_df["fraud_rate_pct"]):
        ax2.annotate(f"{rate:.1f}%", (i, rate), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, fontweight="bold", color="#b23a22")

    ax1.set_xlabel(category_col.replace("_", " ").title(), fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Transaction Count", fontsize=11, fontweight="bold", color="#333333")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold", color="#b23a22")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df[category_col], rotation=rot, ha="right" if rot > 0 else "center", fontsize=10)
    ax1.grid(axis="x")
    ax2.grid(False)

    # Title & Legend
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    out_path = fig_dir / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_all_eda_visualizations(df_dev: pd.DataFrame) -> Dict[str, Path]:
    """
    Orchestrates the generation of all 19 publication-quality EDA figures across:
    - 8 Categorical Visualizations
    - 2 Temporal Visualizations
    - 5 Numerical / Non-linear Visualizations
    - 4 Multi-variable Interaction Heatmaps
    Saves figures to reports/figures/ and tables to reports/tables/.
    """
    base_rate = float(df_dev[TARGET_COL].mean() * 100)
    saved_plots = {}

    # =========================================================================
    # 1. CATEGORICAL ANALYSIS (FIGURES 01 - 08)
    # =========================================================================
    # Figure 1: Target Class Balance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    class_counts = df_dev[TARGET_COL].value_counts()
    class_labels = ["Legitimate (0)", "Fraudulent (1)"]
    palette = ["#2b5c8f", "#d95f02"]

    bars = ax1.bar(class_labels, class_counts, color=palette, width=0.5)
    for b in bars:
        ax1.annotate(f"{b.get_height():,} ({b.get_height()/len(df_dev)*100:.1f}%)",
                     (b.get_x() + b.get_width()/2, b.get_height() + 25),
                     ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Transaction Count by Target Class", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Transaction Count")
    ax1.set_ylim(0, 1800)

    ax2.pie(class_counts, labels=class_labels, colors=palette, autopct="%1.2f%%",
            startangle=140, explode=[0, 0.08], textprops={"fontsize": 11, "fontweight": "bold"})
    ax2.set_title("Proportion of Target Class (Imbalance Ratio 7.25:1)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    f1_path = FIGURES_DIR / "eda_01_target_class_distribution.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    saved_plots["fig_01"] = f1_path

    # Figure 2: Channel
    prof_channel = compute_categorical_risk_profiles(df_dev, "channel")
    prof_channel.to_csv(TABLES_DIR / "eda_channel_risk_profile.csv", index=False)
    saved_plots["fig_02"] = plot_dual_axis_rate_and_volume(
        prof_channel, "channel", "Fraud Rate and Volume across Payment Channels",
        "eda_02_fraud_by_channel.png", base_rate
    )

    # Figure 3: Merchant Category
    prof_merchant = compute_categorical_risk_profiles(df_dev, "merchant_category")
    prof_merchant.to_csv(TABLES_DIR / "eda_merchant_category_risk_profile.csv", index=False)
    saved_plots["fig_03"] = plot_dual_axis_rate_and_volume(
        prof_merchant, "merchant_category", "Fraud Rate and Volume across Merchant Categories",
        "eda_03_fraud_by_merchant_category.png", base_rate, rot=30
    )

    # Figure 4: Authentication Level (including explicit MISSING)
    prof_auth = compute_categorical_risk_profiles(df_dev, "authentication_level", fill_missing="MISSING (Unrecorded)")
    prof_auth.to_csv(TABLES_DIR / "eda_authentication_level_risk_profile.csv", index=False)
    saved_plots["fig_04"] = plot_dual_axis_rate_and_volume(
        prof_auth, "authentication_level", "Fraud Rate by Authentication Protocol Handshake",
        "eda_04_fraud_by_authentication_level.png", base_rate
    )

    # Figure 5: Customer Tier
    prof_tier = compute_categorical_risk_profiles(df_dev, "customer_tier")
    prof_tier.to_csv(TABLES_DIR / "eda_customer_tier_risk_profile.csv", index=False)
    saved_plots["fig_05"] = plot_dual_axis_rate_and_volume(
        prof_tier, "customer_tier", "Fraud Rate and Volume across Customer Relationship Tiers",
        "eda_05_fraud_by_customer_tier.png", base_rate
    )

    # Figure 6: Device Type
    prof_device = compute_categorical_risk_profiles(df_dev, "device_type")
    prof_device.to_csv(TABLES_DIR / "eda_device_type_risk_profile.csv", index=False)
    saved_plots["fig_06"] = plot_dual_axis_rate_and_volume(
        prof_device, "device_type", "Fraud Rate and Volume across Client Device Architectures",
        "eda_06_fraud_by_device_type.png", base_rate
    )

    # Figure 7: Card Presence
    prof_card = compute_categorical_risk_profiles(df_dev, "card_presence")
    prof_card.to_csv(TABLES_DIR / "eda_card_presence_risk_profile.csv", index=False)
    saved_plots["fig_07"] = plot_dual_axis_rate_and_volume(
        prof_card, "card_presence", "Fraud Rate by Physical Card Presence (CNP vs. CP)",
        "eda_07_fraud_by_card_presence.png", base_rate
    )

    # Figure 8: Country
    prof_country = compute_categorical_risk_profiles(df_dev, "country")
    prof_country.to_csv(TABLES_DIR / "eda_country_risk_profile.csv", index=False)
    saved_plots["fig_08"] = plot_dual_axis_rate_and_volume(
        prof_country, "country", "Fraud Rate and Volume across Transaction Origin Countries",
        "eda_08_fraud_by_country.png", base_rate, rot=25
    )

    # =========================================================================
    # 2. TEMPORAL ANALYSIS (FIGURES 09 - 10)
    # =========================================================================
    # Figure 9: Hour (0-23)
    hour_summary = df_dev.groupby("hour")[TARGET_COL].agg(
        volume="count",
        fraud_count="sum"
    ).reset_index()
    hour_summary["fraud_rate_pct"] = (hour_summary["fraud_count"] / hour_summary["volume"] * 100).round(2)
    hour_summary.to_csv(TABLES_DIR / "eda_temporal_hour_risk_profile.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
    ax2 = ax1.twinx()
    ax1.bar(hour_summary["hour"], hour_summary["volume"], color="#90afc7", alpha=0.7, width=0.6, label="Hourly Volume")
    ax2.plot(hour_summary["hour"], hour_summary["fraud_rate_pct"], color="#b23a22", marker="s", linewidth=2.5, label="Fraud Rate (%)")
    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Dev Base Rate ({base_rate:.2f}%)")

    # Highlight nocturnal danger zone (02:00 - 05:00)
    ax2.axvspan(1.5, 5.5, color="#f4a582", alpha=0.3, label="Nocturnal Risk Surge Window (02-05h)")

    ax1.set_xlabel("Transaction Hour of Day (0 - 23)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Transaction Count", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold", color="#b23a22")
    ax1.set_xticks(range(24))
    plt.title("Diurnal Fraud Vulnerability Pattern: Nocturnal Off-Peak Surge", fontsize=12, fontweight="bold", pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    plt.tight_layout()
    f9_path = FIGURES_DIR / "eda_09_fraud_by_hour.png"
    plt.savefig(f9_path, dpi=300)
    plt.close()
    saved_plots["fig_09"] = f9_path

    # Figure 10: Day of Week (0-6)
    dow_names = {0: "Mon (0)", 1: "Tue (1)", 2: "Wed (2)", 3: "Thu (3)", 4: "Fri (4)", 5: "Sat (5)", 6: "Sun (6)"}
    dow_summary = df_dev.groupby("day_of_week")[TARGET_COL].agg(
        volume="count",
        fraud_count="sum"
    ).reset_index()
    dow_summary["day_name"] = dow_summary["day_of_week"].map(dow_names)
    dow_summary["fraud_rate_pct"] = (dow_summary["fraud_count"] / dow_summary["volume"] * 100).round(2)
    dow_summary.to_csv(TABLES_DIR / "eda_temporal_day_of_week_risk_profile.csv", index=False)
    saved_plots["fig_10"] = plot_dual_axis_rate_and_volume(
        dow_summary, "day_name", "Fraud Rate and Volume across Days of the Week",
        "eda_10_fraud_by_day_of_week.png", base_rate
    )

    # =========================================================================
    # 3. NUMERICAL & NON-LINEAR ANALYSIS (FIGURES 11 - 15)
    # =========================================================================
    # Figure 11: Transaction Amount
    # Plot histogram / KDE and binned rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    sns.histplot(data=df_dev, x="transaction_amount", hue=TARGET_COL, palette=["#2b5c8f", "#d95f02"],
                 log_scale=True, bins=30, common_norm=False, stat="density", ax=ax1)
    ax1.set_title("Log-Scaled Transaction Amount Density by Class", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Transaction Amount ($ - Log Scale)")
    ax1.legend(title="Class", labels=["Fraudulent (1)", "Legitimate (0)"])

    # Quantile bins
    amt_bins = compute_binned_numerical_risk_profiles(df_dev, "transaction_amount", n_bins=5)
    amt_bins.to_csv(TABLES_DIR / "eda_binned_transaction_amount.csv", index=False)
    ax2.bar(range(len(amt_bins)), amt_bins["fraud_rate_pct"], color="#d95f02", alpha=0.85, width=0.55)
    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(amt_bins["fraud_rate_pct"]):
        ax2.annotate(f"{r:.1f}%\n(n={amt_bins.loc[i, 'volume']})", (i, r), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(amt_bins)))
    ax2.set_xticklabels(amt_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax2.set_title("Empirical Fraud Rate across Amount Quintiles", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_xlabel("Amount Quintile ($)")
    ax2.legend()
    plt.tight_layout()
    f11_path = FIGURES_DIR / "eda_11_transaction_amount_analysis.png"
    plt.savefig(f11_path, dpi=300)
    plt.close()
    saved_plots["fig_11"] = f11_path

    # Figure 12: Velocity (1h and 24h)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    v1_summary = df_dev.groupby("transactions_last_1h")[TARGET_COL].agg(volume="count", fraud_count="sum").reset_index()
    v1_summary["fraud_rate_pct"] = (v1_summary["fraud_count"] / v1_summary["volume"] * 100).round(2)
    # Filter bins with volume >= 5 for stability
    v1_stable = v1_summary[v1_summary["volume"] >= 5]
    ax1.plot(v1_stable["transactions_last_1h"], v1_stable["fraud_rate_pct"], color="#b23a22", marker="o", linewidth=2)
    ax1.bar(v1_stable["transactions_last_1h"], v1_stable["volume"], color="#90afc7", alpha=0.4, width=0.5)
    ax1.axhline(base_rate, color="#2b5c8f", linestyle="--")
    ax1.set_title("Fraud Rate vs. Transactions Last 1 Hour (1h Velocity)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Transactions in Prior 1 Hour")
    ax1.set_ylabel("Fraud Rate (%)")

    # 24h Velocity Binned
    v24_bins = compute_binned_numerical_risk_profiles(df_dev, "transactions_last_24h", n_bins=5)
    ax2.bar(range(len(v24_bins)), v24_bins["fraud_rate_pct"], color="#d95f02", alpha=0.85, width=0.55)
    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(v24_bins["fraud_rate_pct"]):
        ax2.annotate(f"{r:.1f}%\n(n={v24_bins.loc[i, 'volume']})", (i, r), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(v24_bins)))
    ax2.set_xticklabels(v24_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax2.set_title("Fraud Rate vs. Transactions Last 24 Hours (24h Velocity)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.legend()
    plt.tight_layout()
    f12_path = FIGURES_DIR / "eda_12_velocity_1h_24h_analysis.png"
    plt.savefig(f12_path, dpi=300)
    plt.close()
    saved_plots["fig_12"] = f12_path

    # Figure 13: Distance from Usual Location
    dist_bins = compute_binned_numerical_risk_profiles(df_dev, "distance_from_usual_location_km", n_bins=5)
    dist_bins.to_csv(TABLES_DIR / "eda_binned_distance.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2 = ax1.twinx()
    x = np.arange(len(dist_bins))
    ax1.bar(x, dist_bins["volume"], color="#90afc7", alpha=0.7, width=0.5, label="Transaction Volume")
    ax2.plot(x, dist_bins["fraud_rate_pct"], color="#b23a22", marker="D", linewidth=2.5, label="Fraud Rate (%)")
    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(dist_bins["fraud_rate_pct"]):
        ax2.annotate(f"{r:.1f}%", (i, r), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#b23a22")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dist_bins["bin"], rotation=15, ha="right", fontsize=10)
    ax1.set_xlabel("Distance Quintiles (km)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Volume", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold", color="#b23a22")
    plt.title("Spatial Anomaly Analysis: Distance from Usual Location vs. Fraud Rate", fontsize=12, fontweight="bold", pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    f13_path = FIGURES_DIR / "eda_13_distance_from_usual_location.png"
    plt.savefig(f13_path, dpi=300)
    plt.close()
    saved_plots["fig_13"] = f13_path

    # Figure 14: Account Age and Device Age Days
    acc_bins = compute_binned_numerical_risk_profiles(df_dev, "account_age_days", n_bins=5)
    dev_age_bins = compute_binned_numerical_risk_profiles(df_dev, "device_age_days", n_bins=5)
    acc_bins.to_csv(TABLES_DIR / "eda_binned_account_age.csv", index=False)
    dev_age_bins.to_csv(TABLES_DIR / "eda_binned_device_age.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.bar(range(len(acc_bins)), acc_bins["fraud_rate_pct"], color="#41b6c4", width=0.55)
    ax1.axhline(base_rate, color="red", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(acc_bins["fraud_rate_pct"]):
        ax1.annotate(f"{r:.1f}%\n(n={acc_bins.loc[i, 'volume']})", (i, r), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
    ax1.set_xticks(range(len(acc_bins)))
    ax1.set_xticklabels(acc_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax1.set_title("Fraud Rate vs. Account Age (Days)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fraud Rate (%)")
    ax1.legend()

    ax2.bar(range(len(dev_age_bins)), dev_age_bins["fraud_rate_pct"], color="#2c7fb8", width=0.55)
    ax2.axhline(base_rate, color="red", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(dev_age_bins["fraud_rate_pct"]):
        ax2.annotate(f"{r:.1f}%\n(n={dev_age_bins.loc[i, 'volume']})", (i, r), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
    ax2.set_xticks(range(len(dev_age_bins)))
    ax2.set_xticklabels(dev_age_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax2.set_title("Fraud Rate vs. Device Age (Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.legend()
    plt.tight_layout()
    f14_path = FIGURES_DIR / "eda_14_account_and_device_age_analysis.png"
    plt.savefig(f14_path, dpi=300)
    plt.close()
    saved_plots["fig_14"] = f14_path

    # Figure 15: Merchant Risk Score and IP Risk Score
    merch_bins = compute_binned_numerical_risk_profiles(df_dev, "merchant_risk_score", n_bins=5)
    ip_bins = compute_binned_numerical_risk_profiles(df_dev, "ip_risk_score", n_bins=5)
    merch_bins.to_csv(TABLES_DIR / "eda_binned_merchant_risk_score.csv", index=False)
    ip_bins.to_csv(TABLES_DIR / "eda_binned_ip_risk_score.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(range(len(merch_bins)), merch_bins["fraud_rate_pct"], marker="o", color="#b23a22", linewidth=2.5)
    ax1.bar(range(len(merch_bins)), merch_bins["volume"], color="#90afc7", alpha=0.4, width=0.5)
    ax1.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(merch_bins["fraud_rate_pct"]):
        ax1.annotate(f"{r:.1f}%", (i, r), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold")
    ax1.set_xticks(range(len(merch_bins)))
    ax1.set_xticklabels(merch_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax1.set_title("Fraud Rate vs. Merchant Risk Score Quintiles", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fraud Rate (%)")
    ax1.legend()

    ax2.plot(range(len(ip_bins)), ip_bins["fraud_rate_pct"], marker="s", color="#b23a22", linewidth=2.5)
    ax2.bar(range(len(ip_bins)), ip_bins["volume"], color="#90afc7", alpha=0.4, width=0.5)
    ax2.axhline(base_rate, color="#2b5c8f", linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
    for i, r in enumerate(ip_bins["fraud_rate_pct"]):
        ax2.annotate(f"{r:.1f}%", (i, r), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold")
    ax2.set_xticks(range(len(ip_bins)))
    ax2.set_xticklabels(ip_bins["bin"], rotation=20, ha="right", fontsize=9)
    ax2.set_title("Fraud Rate vs. IP Risk Score Quintiles", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.legend()
    plt.tight_layout()
    f15_path = FIGURES_DIR / "eda_15_risk_scores_analysis.png"
    plt.savefig(f15_path, dpi=300)
    plt.close()
    saved_plots["fig_15"] = f15_path

    # =========================================================================
    # 4. INTERACTION HEATMAPS & MULTI-FEATURE ANALYSIS (FIGURES 16 - 19)
    # =========================================================================
    # Figure 16: Interaction Heatmap: new_device x distance_from_usual_location_km
    dist_cat = pd.qcut(df_dev["distance_from_usual_location_km"].dropna(), q=4, labels=["Near (<12km)", "Moderate (12-32km)", "Far (32-69km)", "Extreme (>69km)"])
    temp_16 = pd.DataFrame({
        "new_device": df_dev["new_device"].map({0: "Existing Device (0)", 1: "New Device (1)"}),
        "distance_tier": dist_cat,
        TARGET_COL: df_dev[TARGET_COL]
    })
    pivot_16_rate = temp_16.pivot_table(index="new_device", columns="distance_tier", values=TARGET_COL, aggfunc=lambda y: np.mean(y)*100)
    pivot_16_vol = temp_16.pivot_table(index="new_device", columns="distance_tier", values=TARGET_COL, aggfunc="count")
    pivot_16_rate.to_csv(TABLES_DIR / "eda_interaction_new_device_distance_rate.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    annot_matrix = np.empty(pivot_16_rate.shape, dtype=object)
    for i in range(pivot_16_rate.shape[0]):
        for j in range(pivot_16_rate.shape[1]):
            annot_matrix[i, j] = f"{pivot_16_rate.iloc[i, j]:.1f}%\n(n={pivot_16_vol.iloc[i, j]})"

    sns.heatmap(pivot_16_rate, annot=annot_matrix, fmt="", cmap="YlOrRd", cbar_kws={'label': 'Fraud Rate (%)'}, ax=ax, linewidths=1.2)
    ax.set_title("Interaction Heatmap 1: New Device x Spatial Distance Tier", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Spatial Distance Tier (km)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Device Status", fontsize=11, fontweight="bold")
    plt.tight_layout()
    f16_path = FIGURES_DIR / "eda_16_interaction_new_device_x_distance.png"
    plt.savefig(f16_path, dpi=300)
    plt.close()
    saved_plots["fig_16"] = f16_path

    # Figure 17: Interaction Heatmap: transaction_amount x merchant_risk_score
    amt_cat = pd.qcut(df_dev["transaction_amount"].dropna(), q=3, labels=["Low ($<33)", "Mid ($33-85)", "High ($>85)"])
    merch_cat = pd.qcut(df_dev["merchant_risk_score"].dropna(), q=3, labels=["Low Risk (<0.22)", "Mid Risk (0.22-0.45)", "High Risk (>0.45)"])
    temp_17 = pd.DataFrame({
        "amount_tier": amt_cat,
        "merchant_risk_tier": merch_cat,
        TARGET_COL: df_dev[TARGET_COL]
    })
    pivot_17_rate = temp_17.pivot_table(index="amount_tier", columns="merchant_risk_tier", values=TARGET_COL, aggfunc=lambda y: np.mean(y)*100)
    pivot_17_vol = temp_17.pivot_table(index="amount_tier", columns="merchant_risk_tier", values=TARGET_COL, aggfunc="count")
    pivot_17_rate.to_csv(TABLES_DIR / "eda_interaction_amount_merchant_risk_rate.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    annot_matrix_17 = np.empty(pivot_17_rate.shape, dtype=object)
    for i in range(pivot_17_rate.shape[0]):
        for j in range(pivot_17_rate.shape[1]):
            annot_matrix_17[i, j] = f"{pivot_17_rate.iloc[i, j]:.1f}%\n(n={pivot_17_vol.iloc[i, j]})"

    sns.heatmap(pivot_17_rate, annot=annot_matrix_17, fmt="", cmap="YlOrRd", cbar_kws={'label': 'Fraud Rate (%)'}, ax=ax, linewidths=1.2)
    ax.set_title("Interaction Heatmap 2: Transaction Amount x Merchant Risk Tier", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Merchant Risk Tier", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transaction Amount Tier", fontsize=11, fontweight="bold")
    plt.tight_layout()
    f17_path = FIGURES_DIR / "eda_17_interaction_amount_x_merchant_risk.png"
    plt.savefig(f17_path, dpi=300)
    plt.close()
    saved_plots["fig_17"] = f17_path

    # Figure 18: Interaction Heatmap: authentication_level x channel
    auth_filled = df_dev["authentication_level"].fillna("MISSING")
    temp_18 = pd.DataFrame({
        "authentication_level": auth_filled,
        "channel": df_dev["channel"],
        TARGET_COL: df_dev[TARGET_COL]
    })
    pivot_18_rate = temp_18.pivot_table(index="authentication_level", columns="channel", values=TARGET_COL, aggfunc=lambda y: np.mean(y)*100)
    pivot_18_vol = temp_18.pivot_table(index="authentication_level", columns="channel", values=TARGET_COL, aggfunc="count")
    pivot_18_rate.to_csv(TABLES_DIR / "eda_interaction_auth_level_channel_rate.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    annot_matrix_18 = np.empty(pivot_18_rate.shape, dtype=object)
    for i in range(pivot_18_rate.shape[0]):
        for j in range(pivot_18_rate.shape[1]):
            val = pivot_18_rate.iloc[i, j]
            cnt = pivot_18_vol.iloc[i, j] if not np.isnan(pivot_18_vol.iloc[i, j]) else 0
            annot_matrix_18[i, j] = f"{val:.1f}%\n(n={cnt})" if not np.isnan(val) else "N/A\n(n=0)"

    sns.heatmap(pivot_18_rate.fillna(0), annot=annot_matrix_18, fmt="", cmap="YlOrRd", cbar_kws={'label': 'Fraud Rate (%)'}, ax=ax, linewidths=1.2)
    ax.set_title("Interaction Heatmap 3: Authentication Protocol x Payment Channel", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Payment Channel", fontsize=11, fontweight="bold")
    ax.set_ylabel("Authentication Protocol Handshake", fontsize=11, fontweight="bold")
    plt.tight_layout()
    f18_path = FIGURES_DIR / "eda_18_interaction_auth_level_x_channel.png"
    plt.savefig(f18_path, dpi=300)
    plt.close()
    saved_plots["fig_18"] = f18_path

    # Figure 19: Interaction Heatmap: failed_logins_24h x ip_risk_score
    login_bins = pd.cut(df_dev["failed_logins_24h"], bins=[-0.5, 0.5, 2.5, 10.5], labels=["0 Logins", "1-2 Logins", "3+ Logins"])
    ip_cat = pd.qcut(df_dev["ip_risk_score"].dropna(), q=3, labels=["Low IP Risk", "Mid IP Risk", "High IP Risk"])
    temp_19 = pd.DataFrame({
        "failed_logins": login_bins,
        "ip_risk": ip_cat,
        TARGET_COL: df_dev[TARGET_COL]
    })
    pivot_19_rate = temp_19.pivot_table(index="failed_logins", columns="ip_risk", values=TARGET_COL, aggfunc=lambda y: np.mean(y)*100)
    pivot_19_vol = temp_19.pivot_table(index="failed_logins", columns="ip_risk", values=TARGET_COL, aggfunc="count")
    pivot_19_rate.to_csv(TABLES_DIR / "eda_interaction_failed_logins_ip_risk_rate.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    annot_matrix_19 = np.empty(pivot_19_rate.shape, dtype=object)
    for i in range(pivot_19_rate.shape[0]):
        for j in range(pivot_19_rate.shape[1]):
            annot_matrix_19[i, j] = f"{pivot_19_rate.iloc[i, j]:.1f}%\n(n={pivot_19_vol.iloc[i, j]})"

    sns.heatmap(pivot_19_rate, annot=annot_matrix_19, fmt="", cmap="YlOrRd", cbar_kws={'label': 'Fraud Rate (%)'}, ax=ax, linewidths=1.2)
    ax.set_title("Interaction Heatmap 4: Failed Logins (24h) x IP Risk Score Tier", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("IP Risk Score Tier", fontsize=11, fontweight="bold")
    ax.set_ylabel("Failed Logins (Past 24h)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    f19_path = FIGURES_DIR / "eda_19_interaction_failed_logins_x_ip_risk.png"
    plt.savefig(f19_path, dpi=300)
    plt.close()
    saved_plots["fig_19"] = f19_path

    return saved_plots
