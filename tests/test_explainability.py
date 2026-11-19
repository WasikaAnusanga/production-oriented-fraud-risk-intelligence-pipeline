"""
Unit tests for Stage 9: Explainable AI (src/explainability.py).

Tests:
- get_pipeline_feature_names helper.
- Global feature importance calculation (TreeSHAP and Gini).
- Feature importance stability computation across calibrator folds.
- Local explanation generation and SHAP waterfall calculation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from src.explainability import (
    clean_feature_name,
    get_pipeline_feature_names,
    compute_global_feature_importance,
    analyze_feature_importance_stability,
    generate_local_explanations,
)
from src.feature_engineering import AdvancedFraudFeatureEngineer
from src.preprocessing import build_preprocessor, ALL_NUMERIC_FEATURES
from config.settings import CATEGORICAL_PREDICTORS, RAW_DATA_PATH
from src.data_splitting import get_partition_data


def test_clean_feature_name():
    assert clean_feature_name("num__transaction_amount") == "transaction_amount"
    assert clean_feature_name("cat__channel_POS") == "channel_POS"
    assert clean_feature_name("remainder__id") == "id"


def test_explainability_pipeline_execution():
    # Load small slice of real development partition
    X_dev, y_dev, ids_dev = get_partition_data(partition="development")
    sample_X = X_dev.head(60)
    sample_y = y_dev.head(60)
    sample_ids = ids_dev.head(60)

    # Build standard pipeline
    preprocessor = build_preprocessor(
        numeric_features=ALL_NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_PREDICTORS,
        scale_numeric=False,
    )
    pipe = Pipeline([
        ("fe", AdvancedFraudFeatureEngineer(include_original=True)),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)),
    ])
    pipe.fit(sample_X, sample_y)

    # 1. Global importance test
    df_imp, shap_mat, exp_obj, X_trans = compute_global_feature_importance(
        pipeline=pipe,
        X_sample=sample_X,
        y_sample=sample_y,
        n_permutations=2,
    )
    assert len(df_imp) > 0
    assert "mean_abs_shap" in df_imp.columns
    assert "gini_importance" in df_imp.columns
    assert shap_mat.shape[0] == len(sample_X)

    # 2. Calibrated stability test
    cal_clf = CalibratedClassifierCV(estimator=pipe, cv=3)
    cal_clf.fit(sample_X, sample_y)
    stab_df = analyze_feature_importance_stability(cal_clf, sample_X, sample_y, top_k=5)
    assert len(stab_df) == 5
    assert "cv_pct_stability" in stab_df.columns

    # 3. Local explanations test
    local_df, plots = generate_local_explanations(
        pipeline=pipe,
        calibrated_pipeline=cal_clf,
        X_cohort=sample_X,
        y_cohort=sample_y,
        ids_cohort=sample_ids,
        output_dir="reports/figures",
    )
    assert len(local_df) == 5
    assert "top_risk_increasing_factors" in local_df.columns
    assert "plain_language_explanation" in local_df.columns
