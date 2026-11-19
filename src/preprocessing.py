"""
Preprocessing Pipeline Module.

Builds leakage-free scikit-learn ColumnTransformers with imputation,
scaling, one-hot encoding, and missing indicator flags.
Designed to integrate directly with feature engineering transformers
and imbalanced-learn pipelines.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.settings import NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS

# Complete list of numeric features after AdvancedFraudFeatureEngineer
ENGINEERED_NUMERIC_FEATURES = [
    "log_transaction_amount",
    "log_distance",
    "velocity_ratio",
    "previous_23h_transactions",
    "amount_per_recent_transaction",
    "combined_merchant_ip_risk",
    "merchant_ip_risk_difference",
    "amount_merchant_risk_interaction",
    "new_device_distance_interaction",
    "failed_login_ip_risk",
    "account_device_age_ratio",
    "email_account_age_ratio",
    "hour_sin",
    "hour_cos",
    "billing_distance_interaction",
    "chargeback_velocity_interaction",
    "login_velocity_interaction",
    "normalized_amount",
]

ALL_NUMERIC_FEATURES = NUMERIC_PREDICTORS + ENGINEERED_NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
    add_indicator: bool = True,
) -> ColumnTransformer:
    """
    Constructs a leakage-free ColumnTransformer:
    - Numeric: Median imputation with missingness indicators + optional StandardScaler.
    - Categorical: Constant imputation ('MISSING') + OneHotEncoder(handle_unknown='ignore').

    Parameters:
    -----------
    numeric_features : Optional[List[str]]
        List of numeric column names. Defaults to ALL_NUMERIC_FEATURES.
    categorical_features : Optional[List[str]]
        List of categorical column names. Defaults to CATEGORICAL_PREDICTORS.
    scale_numeric : bool
        Whether to apply StandardScaler (required for linear models; optional for tree models).
    add_indicator : bool
        Whether to append MissingIndicator boolean flags for numeric missing values.

    Returns:
    --------
    ColumnTransformer
    """
    if numeric_features is None:
        numeric_features = ALL_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_PREDICTORS

    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
