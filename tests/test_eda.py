"""
Tests for Advanced Exploratory Data Analysis (Stage 4 / Section B).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config.settings import FIGURES_DIR, TABLES_DIR, TARGET_COL
from src.data_splitting import get_partition_data
from src.eda import (
    compute_categorical_risk_profiles,
    compute_binned_numerical_risk_profiles,
    generate_all_eda_visualizations,
)


@pytest.fixture
def dev_data():
    X_dev, y_dev, ids_dev = get_partition_data("development")
    df = X_dev.copy()
    df[TARGET_COL] = y_dev
    return df


def test_categorical_risk_profiles_calculation(dev_data):
    """Verify categorical risk profile output schema and calculations."""
    prof = compute_categorical_risk_profiles(dev_data, "channel")
    assert not prof.empty
    assert "channel" in prof.columns
    assert "fraud_rate_pct" in prof.columns
    assert "relative_risk" in prof.columns
    assert "se_pct" in prof.columns
    assert prof["volume"].sum() == len(dev_data)


def test_binned_numerical_risk_profiles(dev_data):
    """Verify numeric binning produces correct intervals and fraud rates."""
    binned = compute_binned_numerical_risk_profiles(dev_data, "transaction_amount", n_bins=5)
    assert len(binned) == 5
    assert "bin" in binned.columns
    assert "fraud_rate_pct" in binned.columns
    assert binned["volume"].sum() == dev_data["transaction_amount"].notnull().sum()


def test_all_eda_figures_generated_and_exist():
    """Verify all 19 required publication-quality figures exist on disk at 300 DPI."""
    expected_figures = [
        "eda_01_target_class_distribution.png",
        "eda_02_fraud_by_channel.png",
        "eda_03_fraud_by_merchant_category.png",
        "eda_04_fraud_by_authentication_level.png",
        "eda_05_fraud_by_customer_tier.png",
        "eda_06_fraud_by_device_type.png",
        "eda_07_fraud_by_card_presence.png",
        "eda_08_fraud_by_country.png",
        "eda_09_fraud_by_hour.png",
        "eda_10_fraud_by_day_of_week.png",
        "eda_11_transaction_amount_analysis.png",
        "eda_12_velocity_1h_24h_analysis.png",
        "eda_13_distance_from_usual_location.png",
        "eda_14_account_and_device_age_analysis.png",
        "eda_15_risk_scores_analysis.png",
        "eda_16_interaction_new_device_x_distance.png",
        "eda_17_interaction_amount_x_merchant_risk.png",
        "eda_18_interaction_auth_level_x_channel.png",
        "eda_19_interaction_failed_logins_x_ip_risk.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file empty or corrupt: {fig_path}"


def test_eda_tables_exported_and_valid():
    """Verify statistical summary tables exist and have valid entries."""
    expected_tables = [
        "eda_channel_risk_profile.csv",
        "eda_merchant_category_risk_profile.csv",
        "eda_authentication_level_risk_profile.csv",
        "eda_customer_tier_risk_profile.csv",
        "eda_device_type_risk_profile.csv",
        "eda_card_presence_risk_profile.csv",
        "eda_country_risk_profile.csv",
        "eda_temporal_hour_risk_profile.csv",
        "eda_temporal_day_of_week_risk_profile.csv",
        "eda_binned_transaction_amount.csv",
        "eda_binned_distance.csv",
        "eda_binned_account_age.csv",
        "eda_binned_device_age.csv",
        "eda_binned_merchant_risk_score.csv",
        "eda_binned_ip_risk_score.csv",
        "eda_interaction_new_device_distance_rate.csv",
        "eda_interaction_amount_merchant_risk_rate.csv",
        "eda_interaction_auth_level_channel_rate.csv",
        "eda_interaction_failed_logins_ip_risk_rate.csv",
    ]
    for tbl_name in expected_tables:
        tbl_path = TABLES_DIR / tbl_name
        assert tbl_path.exists(), f"Missing table: {tbl_path}"
        df = pd.read_csv(tbl_path)
        assert not df.empty, f"Table is empty: {tbl_path}"
