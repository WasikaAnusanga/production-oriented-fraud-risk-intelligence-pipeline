"""
Unit and integration tests for Fraud Decision Engine & Operational Triage.

Verifies:
- All predicted probabilities fall strictly in [0, 1].
- Every transaction receives exactly one deterministic action (APPROVE, MANUAL REVIEW, BLOCK).
- Decision thresholds work correctly.
- Monotonicity: Increasing risk does not lead to a less severe action under a fixed threshold policy.
- Zero target requirement: Inference operates seamlessly without the 'fraud' target column.
- Pipeline serialization consistency: Model outputs match perfectly after joblib save and reload.
- Robustness: Handles missing numerical fields and unseen categorical levels without crashing.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import joblib

from src.fraud_decision_engine import FraudDecisionEngine, DEFAULT_REVIEW_THRESHOLD, DEFAULT_BLOCK_THRESHOLD
from src.data_splitting import get_partition_data
from config.settings import ALL_PREDICTORS


@pytest.fixture(scope="module")
def engine():
    return FraudDecisionEngine(
        model_path="04_calibrated_model.joblib",
        review_threshold=0.10,
        block_threshold=0.36,
    )


@pytest.fixture(scope="module")
def sample_batch():
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    batch = X_val.head(25).copy()
    batch["transaction_id"] = ids_val.head(25).values
    return batch


def test_predict_transaction_risk_validity(engine, sample_batch):
    # Verify inference without target 'fraud'
    assert "fraud" not in sample_batch.columns

    results = engine.predict_transaction_risk(sample_batch, include_explanations=True)

    # 1. Row count match
    assert len(results) == len(sample_batch)

    # 2. Probability bounds [0, 1]
    probs = results["fraud_probability"].values
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert not np.isnan(probs).any()

    # 3. Deterministic single action per transaction
    valid_decisions = {"APPROVE", "MANUAL REVIEW", "BLOCK"}
    assert set(results["operational_decision"].unique()).issubset(valid_decisions)

    # 4. Correct threshold routing
    for _, row in results.iterrows():
        p = row["fraud_probability"]
        dec = row["operational_decision"]
        if p < 0.10:
            assert dec == "APPROVE"
        elif p < 0.36:
            assert dec == "MANUAL REVIEW"
        else:
            assert dec == "BLOCK"

    # 5. Explanations populated
    assert "top_risk_factors" in results.columns
    assert "plain_language_reason" in results.columns
    assert results["plain_language_reason"].str.len().gt(10).all()


def test_threshold_monotonicity(engine):
    """Verify that increasing probability strictly produces non-decreasing severity: APPROVE -> REVIEW -> BLOCK."""
    test_probs = [0.01, 0.05, 0.099, 0.10, 0.20, 0.359, 0.36, 0.50, 0.95]
    severity_rank = {"APPROVE": 1, "MANUAL REVIEW": 2, "BLOCK": 3}

    assigned_severities = []
    for p in test_probs:
        action = engine.route_transaction(p)
        # Normalize if underscore
        clean_action = action.replace("_", " ")
        assigned_severities.append(severity_rank[clean_action])

    # Assert non-decreasing order
    for i in range(len(assigned_severities) - 1):
        assert assigned_severities[i] <= assigned_severities[i + 1]


def test_handling_missing_fields_and_unseen_categories(engine, sample_batch):
    """Verify pipeline gracefully handles nulls in imputable fields and unseen categorical levels."""
    corrupted_batch = sample_batch.copy()

    # Inject extreme missingness in multiple columns
    corrupted_batch.loc[0, "transaction_amount"] = np.nan
    corrupted_batch.loc[1, "merchant_risk_score"] = np.nan
    corrupted_batch.loc[2, "distance_from_usual_location_km"] = np.nan
    corrupted_batch.loc[3, "authentication_level"] = np.nan

    # Inject unseen categorical levels
    corrupted_batch.loc[0, "channel"] = "UNSEEN_DRONE_CHANNEL"
    corrupted_batch.loc[1, "country"] = "ZZ_ATLANTIS"
    corrupted_batch.loc[2, "device_type"] = "QUANTUM_GLASSES"
    corrupted_batch.loc[3, "customer_tier"] = "TIER_99"

    # Must execute cleanly without exception
    results = engine.predict_transaction_risk(corrupted_batch, include_explanations=False)
    assert len(results) == len(corrupted_batch)
    assert not results["fraud_probability"].isnull().any()
    assert (results["fraud_probability"] >= 0.0).all()


def test_model_serialization_consistency(engine, sample_batch):
    """Verify inference results remain identical after saving and reloading the calibrated pipeline."""
    initial_preds = engine.predict_transaction_risk(sample_batch, include_explanations=False)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        joblib.dump(engine.model, tmp_path)
        reloaded_engine = FraudDecisionEngine(model_path=tmp_path)
        reloaded_preds = reloaded_engine.predict_transaction_risk(sample_batch, include_explanations=False)

        np.testing.assert_allclose(
            initial_preds["fraud_probability"].values,
            reloaded_preds["fraud_probability"].values,
            rtol=1e-5,
            atol=1e-5,
        )
        assert (initial_preds["operational_decision"] == reloaded_preds["operational_decision"]).all()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_required_column_raises_error(engine, sample_batch):
    """Ensure missing core predictor raises ValueError."""
    broken_batch = sample_batch.drop(columns=["transaction_amount"])
    with pytest.raises(ValueError, match="missing required predictor columns"):
        engine.predict_transaction_risk(broken_batch)
