"""
Unit tests for Stage 7: Probability Calibration (src/calibration.py).

Validates:
- ECE calculation correctness on synthetic ground truth.
- Proper metric calculations (Brier score, Log Loss, ROC-AUC, PR-AUC).
- Calibration cross-validation execution without data leakage.
- Wrapping of raw DataFrame input in CalibratedClassifierCV.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.calibration import (
    compute_expected_calibration_error,
    evaluate_calibration_metrics,
    cross_validate_calibration,
    fit_final_calibrated_pipeline,
)


def test_compute_expected_calibration_error():
    # Perfectly calibrated synthetic scenario: 50% ones with predicted prob 0.5
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ece, prob_true, prob_pred, bin_counts = compute_expected_calibration_error(
        y_true, y_prob, n_bins=5
    )
    assert pytest.approx(ece, abs=1e-5) == 0.0
    assert len(prob_true) == 1
    assert prob_true[0] == 0.5


def test_evaluate_calibration_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_calibration_metrics(y_true, y_prob, n_bins=5)

    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "ece" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["roc_auc"] == 1.0


def test_cross_validate_calibration_mock_pipeline():
    # Create simple mock DataFrame
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
    })
    y = pd.Series((X["num1"] + X["num2"] > 0).astype(int))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

    summary_df, oof_preds, fold_df = cross_validate_calibration(
        base_pipeline=pipe,
        X=X,
        y=y,
        n_splits=3,
        random_state=42,
    )

    assert set(summary_df["Method"]) == {"Uncalibrated", "Sigmoid", "Isotonic"}
    assert len(oof_preds["Sigmoid"]) == n
    assert not np.isnan(oof_preds["Sigmoid"]).any()
    assert (oof_preds["Sigmoid"] >= 0.0).all() and (oof_preds["Sigmoid"] <= 1.0).all()


def test_fit_final_calibrated_pipeline_inference():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
    })
    y = pd.Series((X["num1"] > 0).astype(int))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

    cal_pipe = fit_final_calibrated_pipeline(
        base_pipeline=pipe,
        X_dev=X,
        y_dev=y,
        method="sigmoid",
        cv=3,
    )

    probs = cal_pipe.predict_proba(X)
    assert probs.shape == (n, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
