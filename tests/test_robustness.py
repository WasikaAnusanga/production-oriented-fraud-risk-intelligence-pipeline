"""
Unit tests for Stage 11: Robustness and Stress Testing (src/robustness.py).

Validates:
- Scenario generation producing 8 distinct stressed datasets.
- Logical count constraint preservation (24h >= 1h) under velocity surges.
- Cohort performance and drift calculations.
- Execution of development fold stability evaluation.
"""

import pytest
import numpy as np
import pandas as pd

from src.robustness import (
    generate_stress_scenarios,
    evaluate_cohort_performance,
    evaluate_development_fold_stability,
)
from src.data_splitting import get_partition_data
from src.cost_optimization import load_business_cost_config
from src.fraud_decision_engine import FraudDecisionEngine


@pytest.fixture(scope="module")
def sample_data():
    X_val, y_val, ids_val = get_partition_data(partition="policy_validation")
    return X_val.head(50), y_val.head(50)


@pytest.fixture(scope="module")
def engine():
    return FraudDecisionEngine(model_path="04_calibrated_model.joblib")


@pytest.fixture(scope="module")
def scenario_cfg():
    cfg = load_business_cost_config("config/business_costs.yaml")
    return cfg["scenarios"]["scenario_b_balanced"]


def test_generate_stress_scenarios(sample_data):
    X, y = sample_data
    scenarios = generate_stress_scenarios(X, random_state=42)

    assert len(scenarios) == 8
    assert "Scenario 1: Amount +25%" in scenarios
    assert "Scenario 3: Velocity Surge" in scenarios
    assert "Scenario 8: Unseen Categorical Drift" in scenarios

    # Check constraint in Scenario 3: 24h >= 1h
    s3 = scenarios["Scenario 3: Velocity Surge"]
    assert (s3["transactions_last_24h"] >= s3["transactions_last_1h"]).all()


def test_evaluate_cohort_performance(engine, sample_data, scenario_cfg):
    X, y = sample_data
    res = evaluate_cohort_performance(
        calibrated_model=engine.model,
        X=X,
        y=y,
        scenario_cfg=scenario_cfg,
    )

    assert "mean_probability" in res
    assert "pct_approve" in res
    assert "pct_review" in res
    assert "pct_block" in res
    assert "cost_per_1k" in res
    assert 0.0 <= res["mean_probability"] <= 1.0
    assert pytest.approx(res["pct_approve"] + res["pct_review"] + res["pct_block"], abs=1e-4) == 100.0


def test_evaluate_development_fold_stability(engine, sample_data, scenario_cfg):
    X, y = sample_data
    df_folds = evaluate_development_fold_stability(
        calibrated_model=engine.model,
        X_dev=X,
        y_dev=y,
        scenario_cfg=scenario_cfg,
    )
    assert len(df_folds) == 7  # 5 folds + Mean + Std
    assert "roc_auc" in df_folds.columns
    assert "cost_per_1k" in df_folds.columns
