"""
Tests for Advanced Feature Engineering (Stage 5 / Section C).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

from config.settings import FIGURES_DIR, TABLES_DIR, TARGET_COL
from src.data_splitting import get_partition_data
from src.feature_engineering import AdvancedFraudFeatureEngineer, generate_feature_dictionary


@pytest.fixture
def dev_data():
    X_dev, y_dev, ids_dev = get_partition_data("development")
    return X_dev, y_dev


def test_feature_engineer_fit_transform(dev_data):
    """Verify transformer fits parameters and transforms safely without NaN expansion."""
    X_dev, y_dev = dev_data
    fe = AdvancedFraudFeatureEngineer(include_original=True)
    fe.fit(X_dev)

    # Check fitted parameter
    assert fe.median_transaction_amount_ is not None
    assert fe.median_transaction_amount_ > 0

    # Transform
    X_trans = fe.transform(X_dev)
    assert isinstance(X_trans, pd.DataFrame)
    assert X_trans.shape[0] == X_dev.shape[0]
    # Original (23) + 18 engineered features = 41 columns
    assert X_trans.shape[1] == X_dev.shape[1] + 18

    # Verify get_feature_names_out
    feature_names = fe.get_feature_names_out()
    assert len(feature_names) == 41
    assert "log_transaction_amount" in feature_names
    assert "velocity_ratio" in feature_names
    assert "new_device_distance_interaction" in feature_names


def test_feature_engineer_handles_edge_cases():
    """Verify safe division by zero, null preservation, and negative bounds."""
    fe = AdvancedFraudFeatureEngineer(include_original=False)
    # Synthetic edge cases
    edge_df = pd.DataFrame({
        "transaction_amount": [np.nan, 0.0, -10.0, 100.0],
        "distance_from_usual_location_km": [0.0, -5.0, np.nan, 50.0],
        "transactions_last_1h": [0, 5, 2, 0],
        "transactions_last_24h": [0, 2, 10, 0],  # 1h > 24h edge test
        "merchant_risk_score": [0.0, np.nan, 0.5, 1.0],
        "ip_risk_score": [np.nan, 0.2, 0.8, 0.0],
        "new_device": [0, 1, 0, 1],
        "failed_logins_24h": [0, 3, 0, 1],
        "account_age_days": [0, 100, 500, 1000],
        "device_age_days": [0, 50, 1000, 200],
        "email_age_days": [0, 200, 100, 500],
        "hour": [0, 6, 12, 23],
        "billing_address_mismatch": [0, 1, 0, 1],
        "prior_chargebacks": [0, 1, 0, 2],
        "login_to_transaction_minutes": [np.nan, 0, 15, 60]
    })
    fe.fit(edge_df)
    res = fe.transform(edge_df)

    # 1. No infinite values generated
    assert not np.isinf(res.to_numpy(dtype=float)).any()

    # 2. Velocity ratio bounded and safe
    assert (res["velocity_ratio"] >= 0.0).all()

    # 3. Previous 23h transactions non-negative
    assert (res["previous_23h_transactions"] >= 0.0).all()

    # 4. Harmonic components bounded in [-1, 1]
    assert (-1.0 <= res["hour_sin"]).all() and (res["hour_sin"] <= 1.0).all()
    assert (-1.0 <= res["hour_cos"]).all() and (res["hour_cos"] <= 1.0).all()


def test_joblib_serialization_compatibility(tmp_path, dev_data):
    """Verify transformer can be serialized and deserialized via joblib with identical behavior."""
    X_dev, _ = dev_data
    fe = AdvancedFraudFeatureEngineer(include_original=True)
    fe.fit(X_dev)
    transformed_1 = fe.transform(X_dev)

    save_path = tmp_path / "test_fe.joblib"
    joblib.dump(fe, save_path)
    loaded_fe = joblib.load(save_path)

    transformed_2 = loaded_fe.transform(X_dev)
    pd.testing.assert_frame_equal(transformed_1, transformed_2)


def test_feature_dictionary_contents():
    """Verify feature dictionary documentation table is populated and valid."""
    df_dict = generate_feature_dictionary()
    assert len(df_dict) == 18
    required_cols = [
        "feature_name", "mathematical_formula", "source_columns",
        "fraud_risk_rationale", "prediction_time_availability",
        "missing_value_treatment", "leakage_risk"
    ]
    for col in required_cols:
        assert col in df_dict.columns


def test_feature_engineering_artifacts_exist():
    """Verify all Stage 5 required tables and figures are created."""
    dict_path = TABLES_DIR / "feature_dictionary.csv"
    comp_path = TABLES_DIR / "feature_engineering_comparison.csv"
    fig_path = FIGURES_DIR / "03_feature_engineering_comparison.png"

    assert dict_path.exists()
    assert comp_path.exists()
    assert fig_path.exists()

    comp_df = pd.read_csv(comp_path)
    assert len(comp_df) == 5
    assert "PR_AUC" in comp_df["Metric"].values
