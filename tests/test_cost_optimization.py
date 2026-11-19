"""
Unit tests for Stage 8: Cost Optimization (src/cost_optimization.py).

Tests:
- Cost calculation for individual transactions under APPROVE, REVIEW, BLOCK.
- Policy cost evaluation accuracy.
- Binary and two-threshold grid searches.
- Monotonicity and bounds of thresholds (0 <= review_threshold < block_threshold <= 1).
"""

import pytest
import numpy as np
import pandas as pd

from src.cost_optimization import (
    load_business_cost_config,
    compute_transaction_cost,
    evaluate_policy_costs,
    grid_search_thresholds,
)


@pytest.fixture
def dummy_scenario():
    return {
        "c_fp_block": 50.0,
        "c_review_base": 6.0,
        "c_review_friction": 3.0,
        "c_block_admin": 1.5,
        "fraud_loss_multiplier": 1.0,
        "fraud_loss_fixed_overhead": 10.0,
        "review_leakage_rate": 0.15,
        "fallback_amount": 100.0,
    }


def test_compute_transaction_cost(dummy_scenario):
    # Legitimate approved -> 0 cost
    assert compute_transaction_cost(0, "APPROVE", 200.0, dummy_scenario) == 0.0

    # Fraud approved -> amount * 1.0 + 10.0 = 210.0
    assert compute_transaction_cost(1, "APPROVE", 200.0, dummy_scenario) == 210.0

    # Legitimate reviewed -> 6.0 + 3.0 = 9.0
    assert compute_transaction_cost(0, "REVIEW", 200.0, dummy_scenario) == 9.0

    # Fraud reviewed -> 6.0 + 0.15 * 210.0 = 37.5
    assert compute_transaction_cost(1, "REVIEW", 200.0, dummy_scenario) == 37.5

    # Legitimate blocked -> 50.0 + 1.5 = 51.5
    assert compute_transaction_cost(0, "BLOCK", 200.0, dummy_scenario) == 51.5

    # Fraud blocked -> 1.5
    assert compute_transaction_cost(1, "BLOCK", 200.0, dummy_scenario) == 1.5


def test_evaluate_policy_costs(dummy_scenario):
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([0.05, 0.25, 0.50, 0.85])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])

    # Policy: rev=0.20, blk=0.70
    # idx 0: p=0.05 -> APPROVE, y=0 -> cost=0
    # idx 1: p=0.25 -> REVIEW, y=1 -> cost = 6 + 0.15*(100+10) = 22.5
    # idx 2: p=0.50 -> REVIEW, y=0 -> cost = 6 + 3 = 9.0
    # idx 3: p=0.85 -> BLOCK, y=1 -> cost = 1.5
    # total = 33.0
    res = evaluate_policy_costs(y_true, probs, amounts, 0.20, 0.70, dummy_scenario)
    assert pytest.approx(res["total_cost"]) == 33.0
    assert res["fraud_approved"] == 0
    assert res["fraud_reviewed"] == 1
    assert res["fraud_blocked"] == 1
    assert res["fraud_capture_rate"] == 1.0


def test_grid_search_thresholds(dummy_scenario):
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.7, 0.9])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])

    df_grid, best_cfg = grid_search_thresholds(
        y_true, probs, amounts, dummy_scenario, step=0.1, mode="two_threshold"
    )
    assert len(df_grid) > 0
    assert 0.0 <= best_cfg["review_threshold"] < best_cfg["block_threshold"] <= 1.0
