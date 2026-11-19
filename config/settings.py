"""
Centralized Configuration and Schema Definitions for Fraud Risk Intelligence Pipeline v2.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "transaction_fraud_risk_advanced_dataset.csv"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
INTERMEDIATE_DIR = REPORTS_DIR / "intermediate"
DOCS_DIR = PROJECT_ROOT / "docs"

# Reproducibility
RANDOM_STATE = 42

# Split Proportions (Stratified by Target)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Schema Definitions
IDENTIFIER_COL = "transaction_id"
TARGET_COL = "fraud"

NUMERIC_PREDICTORS = [
    "hour",
    "day_of_week",
    "transaction_amount",
    "distance_from_usual_location_km",
    "transactions_last_1h",
    "transactions_last_24h",
    "account_age_days",
    "failed_logins_24h",
    "prior_chargebacks",
    "device_age_days",
    "merchant_risk_score",
    "ip_risk_score",
    "email_age_days",
    "login_to_transaction_minutes",
    "new_device",
    "billing_address_mismatch",
]

CATEGORICAL_PREDICTORS = [
    "card_presence",
    "channel",
    "country",
    "device_type",
    "merchant_category",
    "authentication_level",
    "customer_tier",
]

ALL_PREDICTORS = NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS
ALL_COLUMNS = [IDENTIFIER_COL] + ALL_PREDICTORS + [TARGET_COL]

# Verified Benchmark Expectations
EXPECTED_ROW_COUNT = 3000
EXPECTED_COL_COUNT = 25
EXPECTED_FRAUD_COUNT = 364
EXPECTED_LEGIT_COUNT = 2636

EXPECTED_MISSING_COUNTS = {
    "authentication_level": 289,
    "distance_from_usual_location_km": 75,
    "merchant_risk_score": 60,
    "login_to_transaction_minutes": 60,
    "transaction_amount": 53,
    "ip_risk_score": 45,
}
