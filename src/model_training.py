"""
Model Training, Cross-Validation, and Imbalanced Learning Module.

Orchestrates:
- Estimator pipeline creation with feature engineering, preprocessing, resampling, and classification.
- Imbalanced learning strategies (Standard, Class-Weighted, RandomOverSampler, SMOTE) strictly inside training folds.
- Hyperparameter tuning via RandomizedSearchCV with StratifiedKFold cross-validation.
- Comprehensive metric evaluation: PR-AUC, ROC-AUC, Balanced Accuracy, Accuracy, Precision, Recall, F1, Confusion Matrix, and fit time.
- Serialization of candidate model pipelines for probability calibration and production inference.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

from config.settings import RANDOM_STATE, NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS
from src.feature_engineering import AdvancedFraudFeatureEngineer
from src.preprocessing import build_preprocessor, ALL_NUMERIC_FEATURES


def create_fraud_pipeline(
    classifier,
    resampler: Optional[Any] = None,
    scale_numeric: bool = True,
    include_feature_engineering: bool = True,
) -> ImbPipeline:
    """
    Constructs an end-to-end leakage-free pipeline using imbalanced-learn Pipeline:
    1. Feature Engineering (AdvancedFraudFeatureEngineer)
    2. ColumnTransformer Preprocessing (imputation, scaling, one-hot encoding)
    3. Resampler (optional: SMOTE, RandomOverSampler, or None)
    4. Classifier Estimator

    Ensures that resampling, imputation, and feature scaling occur strictly inside training folds.
    """
    steps = []

    if include_feature_engineering:
        steps.append(("fe", AdvancedFraudFeatureEngineer(include_original=True)))
        preprocessor = build_preprocessor(
            numeric_features=ALL_NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_PREDICTORS,
            scale_numeric=scale_numeric,
        )
    else:
        preprocessor = build_preprocessor(
            numeric_features=NUMERIC_PREDICTORS,
            categorical_features=CATEGORICAL_PREDICTORS,
            scale_numeric=scale_numeric,
        )

    steps.append(("preprocessor", preprocessor))

    if resampler is not None:
        steps.append(("resampler", resampler))

    steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


def evaluate_pipeline_cv(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Evaluates an end-to-end pipeline across cross-validation folds on the development set.
    Computes all standard metrics, confusion matrices, and collects predicted probabilities.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {
        "pr_auc": [],
        "roc_auc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fit_time": [],
    }

    y_true_all = []
    y_prob_all = []
    total_cm = np.zeros((2, 2), dtype=int)

    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        fit_t = time.time() - t0

        probs = pipeline.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)

        y_true_all.extend(y_val.tolist())
        y_prob_all.extend(probs.tolist())
        total_cm += confusion_matrix(y_val, preds, labels=[0, 1])

        fold_metrics["pr_auc"].append(average_precision_score(y_val, probs))
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        fold_metrics["accuracy"].append(accuracy_score(y_val, preds))
        fold_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, preds))
        fold_metrics["precision"].append(precision_score(y_val, preds, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, preds, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_val, preds, zero_division=0))
        fold_metrics["fit_time"].append(fit_t)

    summary = {}
    for m, vals in fold_metrics.items():
        summary[f"{m}_mean"] = float(np.mean(vals))
        summary[f"{m}_std"] = float(np.std(vals))

    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "confusion_matrix": total_cm,
        "y_true": np.array(y_true_all),
        "y_prob": np.array(y_prob_all),
    }


def run_hyperparameter_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 15,
) -> Dict[str, Any]:
    """
    Executes RandomizedSearchCV for Logistic Regression, Random Forest,
    and HistGradientBoosting with StratifiedKFold CV optimizing PR-AUC.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # 1. Logistic Regression Param Space
    pipe_lr = create_fraud_pipeline(
        classifier=LogisticRegression(solver="saga", max_iter=1000, random_state=RANDOM_STATE),
        scale_numeric=True,
    )
    param_dist_lr = {
        "classifier__C": np.logspace(-2, 2, 20),
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    }
    search_lr = RandomizedSearchCV(
        pipe_lr,
        param_distributions=param_dist_lr,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    search_lr.fit(X, y)
    results["logistic_regression"] = {
        "best_estimator": search_lr.best_estimator_,
        "best_params": search_lr.best_params_,
        "best_score_pr_auc": float(search_lr.best_score_),
        "search_time_sec": float(time.time() - t0),
    }

    # 2. Random Forest Param Space
    pipe_rf = create_fraud_pipeline(
        classifier=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        scale_numeric=False,
    )
    param_dist_rf = {
        "classifier__n_estimators": [100, 150, 200, 250],
        "classifier__max_depth": [6, 10, 14, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    }
    search_rf = RandomizedSearchCV(
        pipe_rf,
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    search_rf.fit(X, y)
    results["random_forest"] = {
        "best_estimator": search_rf.best_estimator_,
        "best_params": search_rf.best_params_,
        "best_score_pr_auc": float(search_rf.best_score_),
        "search_time_sec": float(time.time() - t0),
    }

    # 3. HistGradientBoosting Param Space
    pipe_hgb = create_fraud_pipeline(
        classifier=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        scale_numeric=False,
    )
    param_dist_hgb = {
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "classifier__max_iter": [50, 100, 150, 200],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__min_samples_leaf": [10, 20, 40],
        "classifier__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "classifier__class_weight": [None, "balanced"],
    }
    search_hgb = RandomizedSearchCV(
        pipe_hgb,
        param_distributions=param_dist_hgb,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    search_hgb.fit(X, y)
    results["hist_gradient_boosting"] = {
        "best_estimator": search_hgb.best_estimator_,
        "best_params": search_hgb.best_params_,
        "best_score_pr_auc": float(search_hgb.best_score_),
        "search_time_sec": float(time.time() - t0),
    }

    return results
