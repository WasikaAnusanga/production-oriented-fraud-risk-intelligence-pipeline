"""
Smoke tests for project environment, directory layout, data integrity, and module imports.
"""

from pathlib import Path
import pytest
import pandas as pd

from config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_PATH,
    SPLITS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    INTERMEDIATE_DIR,
    DOCS_DIR,
    RANDOM_STATE,
    EXPECTED_ROW_COUNT,
    EXPECTED_COL_COUNT,
    EXPECTED_FRAUD_COUNT,
    EXPECTED_LEGIT_COUNT,
    EXPECTED_MISSING_COUNTS,
)
from src import (
    data_audit,
    eda,
    feature_engineering,
    preprocessing,
    model_training,
    calibration,
    cost_optimization,
    explainability,
    fraud_decision_engine,
    robustness,
    governance,
)


def test_directory_structure():
    """Verify that all mandatory project directories exist."""
    required_dirs = [
        DATA_DIR,
        DATA_DIR / "raw",
        SPLITS_DIR,
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        INTERMEDIATE_DIR,
        DOCS_DIR,
    ]
    for d in required_dirs:
        assert d.exists() and d.is_dir(), f"Required directory missing: {d}"


def test_raw_dataset_accessible():
    """Verify raw dataset exists and is readable."""
    assert RAW_DATA_PATH.exists(), f"Raw CSV file missing at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert not df.empty, "Dataset is empty"


def test_dataset_forensics_audit():
    """Verify 25 columns, 3000 rows, target balance, and known missing values."""
    audit_results = data_audit.run_full_dataset_audit(RAW_DATA_PATH)
    assert audit_results["overall_pass"] is True, f"Audit failed: {audit_results}"
    assert audit_results["dimensions"]["actual_rows"] == EXPECTED_ROW_COUNT
    assert audit_results["dimensions"]["actual_cols"] == EXPECTED_COL_COUNT
    assert audit_results["target"]["fraud_count"] == EXPECTED_FRAUD_COUNT
    assert audit_results["target"]["legit_count"] == EXPECTED_LEGIT_COUNT
    assert audit_results["missing_values"]["actual_missing"] == EXPECTED_MISSING_COUNTS


def test_module_imports_and_interfaces():
    """Verify all pipeline modules expose expected baseline interfaces."""
    # Settings
    assert RANDOM_STATE == 42

    # Feature Engineering
    engineer = feature_engineering.FraudFeatureEngineer()
    assert hasattr(engineer, "fit") and hasattr(engineer, "transform")

    # Decision Engine
    engine = fraud_decision_engine.FraudDecisionEngine(review_threshold=0.20, block_threshold=0.70)
    assert engine.route_transaction(0.05) == "APPROVE"
    assert engine.route_transaction(0.40) == "MANUAL_REVIEW"
    assert engine.route_transaction(0.85) == "BLOCK"

    # Calibration Metric
    import numpy as np
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = calibration.calculate_expected_calibration_error(y_true, y_prob)
    assert 0.0 <= ece <= 1.0

    # Cost Function
    loss = cost_optimization.compute_financial_loss(y_true, np.array([0, 0, 0, 1]))
    assert "total_financial_loss" in loss
