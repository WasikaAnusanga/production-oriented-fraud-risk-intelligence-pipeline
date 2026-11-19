"""
Tests for Data Forensics & Quality Engineering (Stage 3 / Section A).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config.settings import RAW_DATA_PATH, IDENTIFIER_COL, TARGET_COL
from src.data_splitting import get_partition_data
from src.data_audit import (
    analyze_dataset_structure,
    run_data_quality_checks,
    analyze_missingness_patterns,
    detect_extreme_observations,
    assess_target_leakage,
    build_data_treatment_decisions,
)


@pytest.fixture
def dev_data():
    X_dev, y_dev, ids_dev = get_partition_data("development")
    df = X_dev.copy()
    df[TARGET_COL] = y_dev
    df[IDENTIFIER_COL] = ids_dev
    return df


def test_dataset_structure_profiling(dev_data):
    """Verify structural profiling returns all 25 columns and valid summaries."""
    overview_df, num_summary_df, cat_summary_df = analyze_dataset_structure(dev_data)
    assert len(overview_df) == 25
    assert set(overview_df["column_name"]) == set(dev_data.columns)
    assert len(num_summary_df) == 16
    assert len(cat_summary_df) > 0


def test_data_quality_checks(dev_data):
    """Verify that domain validation rules execute and detect expected patterns."""
    checks_df = run_data_quality_checks(dev_data)
    assert not checks_df.empty

    # Uniqueness & boundaries should have 0 violations
    fail_checks = checks_df[checks_df["status"] == "FAIL"]
    assert len(fail_checks) == 0, f"Critical quality check failed: {fail_checks.to_dict(orient='records')}"


def test_missingness_forensics(dev_data):
    """Verify statistical testing on the 6 incomplete features in development partition."""
    missing_df = analyze_missingness_patterns(dev_data)
    assert len(missing_df) == 6

    # Verify authentication_level missingness is statistically significant
    auth_row = missing_df[missing_df["feature"] == "authentication_level"].iloc[0]
    assert auth_row["is_significant_05"] is True or auth_row["is_significant_05"] == 1
    assert auth_row["fraud_rate_missing_pct"] > auth_row["fraud_rate_observed_pct"]

    # Verify merchant_risk_score missingness is statistically significant (0 fraud)
    merch_row = missing_df[missing_df["feature"] == "merchant_risk_score"].iloc[0]
    assert merch_row["is_significant_05"] is True or merch_row["is_significant_05"] == 1
    assert merch_row["fraud_rate_missing_pct"] == 0.0


def test_extreme_observations_detection(dev_data):
    """Verify outlier detection and zero-deletion justification."""
    extremes_df = detect_extreme_observations(dev_data)
    assert len(extremes_df) >= 7

    # Check that high velocity (transactions_last_1h) outliers have elevated fraud rate
    v_row = extremes_df[extremes_df["feature"] == "transactions_last_1h"].iloc[0]
    assert v_row["fraud_rate_in_outliers_pct"] > 30.0


def test_target_leakage_assessment():
    """Verify leakage audit properly excludes fraud and transaction_id."""
    leakage_df = assess_target_leakage()
    assert len(leakage_df) >= 12

    excluded_features = set(leakage_df[leakage_df["decision"] == "EXCLUDED"]["feature"])
    assert "fraud" in excluded_features
    assert "transaction_id" in excluded_features


def test_data_treatment_decisions():
    """Verify treatment decision table covers all necessary preprocessing rules."""
    treat_df = build_data_treatment_decisions()
    assert len(treat_df) >= 8
    assert "authentication_level (Categorical Missingness: 289 nulls)" in treat_df["feature_or_issue"].values[0]
