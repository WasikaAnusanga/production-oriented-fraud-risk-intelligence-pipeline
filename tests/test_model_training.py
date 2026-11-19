"""
Tests for Baseline Models, Advanced Models, and Imbalanced Learning (Stage 6 / Sections D & G).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

from config.settings import FIGURES_DIR, TABLES_DIR, TARGET_COL
from src.data_splitting import get_partition_data
from src.model_training import create_fraud_pipeline, evaluate_pipeline_cv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


@pytest.fixture
def dev_data():
    X_dev, y_dev, ids_dev = get_partition_data("development")
    return X_dev, y_dev


def test_model_pipeline_construction():
    """Verify that end-to-end pipelines for LR, RF, and HGB are built with correct steps."""
    pipe_lr = create_fraud_pipeline(LogisticRegression(max_iter=100))
    pipe_rf = create_fraud_pipeline(RandomForestClassifier(n_estimators=10), scale_numeric=False)
    pipe_hgb = create_fraud_pipeline(HistGradientBoostingClassifier(max_iter=10), scale_numeric=False)

    for pipe in [pipe_lr, pipe_rf, pipe_hgb]:
        step_names = [name for name, _ in pipe.steps]
        assert "fe" in step_names
        assert "preprocessor" in step_names
        assert "classifier" in step_names


def test_saved_candidate_models_exist():
    """Verify that top candidate pipelines are serialized and can predict probabilities."""
    inter_dir = Path("reports/intermediate")
    hgb_path = inter_dir / "candidate_hgb_pipeline.joblib"
    rf_path = inter_dir / "candidate_rf_pipeline.joblib"
    lr_path = inter_dir / "candidate_lr_pipeline.joblib"

    for p in [hgb_path, rf_path, lr_path]:
        assert p.exists(), f"Missing model checkpoint: {p}"
        model = joblib.load(p)
        assert hasattr(model, "predict_proba"), f"Model at {p} does not support predict_proba"


def test_model_comparison_table_metrics():
    """Verify model comparison table contains all 12 model configurations and required metrics."""
    comp_path = TABLES_DIR / "model_comparison.csv"
    assert comp_path.exists()
    df = pd.read_csv(comp_path)
    assert len(df) == 12
    required_cols = [
        "Model", "PR_AUC_mean", "PR_AUC_std", "ROC_AUC_mean", "ROC_AUC_std",
        "Balanced_Acc_mean", "Accuracy_mean", "Precision_mean", "Recall_mean", "F1_mean", "Fit_Time_sec"
    ]
    for c in required_cols:
        assert c in df.columns

    # Verify PR-AUC values are within reasonable probabilistic bounds
    assert (df["PR_AUC_mean"] >= 0.10).all() and (df["PR_AUC_mean"] <= 1.0).all()


def test_cross_validation_results_table():
    """Verify per-fold results exist for fold stability analysis."""
    cv_path = TABLES_DIR / "cross_validation_results.csv"
    assert cv_path.exists()
    df = pd.read_csv(cv_path)
    # 12 models x 5 folds = 60 rows
    assert len(df) == 60
    assert set(df["Fold"].unique()) == {1, 2, 3, 4, 5}


def test_hyperparameter_results_table():
    """Verify hyperparameter search results table is populated."""
    hyp_path = TABLES_DIR / "hyperparameter_results.csv"
    assert hyp_path.exists()
    df = pd.read_csv(hyp_path)
    assert len(df) == 3
    assert set(df["model"].unique()) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}


def test_evaluation_figures_exist():
    """Verify all 3 Stage 6 publication-quality figures are generated."""
    figs = [
        "04_model_evaluation_pr_roc_curves.png",
        "05_imbalanced_learning_comparison.png",
        "06_cross_validation_fold_variability.png"
    ]
    for f in figs:
        p = FIGURES_DIR / f
        assert p.exists() and p.stat().st_size > 1000
