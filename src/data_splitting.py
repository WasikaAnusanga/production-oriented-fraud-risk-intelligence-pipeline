"""
Data Splitting and Partition Management Module.

Enforces strict evaluation governance:
- 60% Development (Train / Cross-validation / Feature design / Model tuning / Calibration fitting)
- 20% Policy-Validation (Operational threshold tuning / Multi-tier policy selection)
- 20% Independent Final Test (Held-out final evaluation only; outcomes quarantined during development)

Enforces strict stratification by target 'fraud', random_state=42, and guarantees zero partition overlap.
Ensures transaction_id and fraud are isolated from predictive feature matrices.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config.settings import (
    RAW_DATA_PATH,
    SPLITS_DIR,
    IDENTIFIER_COL,
    TARGET_COL,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    ALL_PREDICTORS,
)

MANIFEST_PATH = SPLITS_DIR / "split_manifest.csv"

# Evaluation Boundaries Documentation
EVALUATION_BOUNDARIES_DOC = """
EVALUATION BOUNDARIES AND DATA HYGIENE PROTOCOLS:

1. Development Partition (60%, n=1,800):
   - Scope: Exploratory Data Analysis (EDA), feature engineering discovery, transformer fitting,
     cross-validation fold generation, algorithm selection, hyperparameter tuning, and probability calibration fitting.
   - Leakage Control: All imputers, encoders, and scalers are fitted exclusively within development folds.

2. Policy-Validation Partition (20%, n=600):
   - Scope: Decision threshold search, operational cost surface evaluation, manual review capacity simulations,
     and 3-tier routing rule selection (APPROVE / MANUAL REVIEW / BLOCK).
   - Leakage Control: Models and calibration transformers are strictly frozen prior to exposing policy validation.
     This partition must never be used for feature selection, hyperparameter tuning, or model fitting.

3. Independent Final Test Partition (20%, n=600):
   - Scope: Independent, unbiased audit of the final locked pipeline (model + calibration + decision policy).
   - Leakage Control: Held out and quarantined. Target labels and outcomes are never accessed or visualized
     during any stage of model design, threshold tuning, or policy selection.

METHODOLOGICAL CONSTRAINTS ON SPLITTING:
- Why Genuine Chronological Split is Unavailable:
  The dataset provides only cyclic temporal components ('hour' [0-23] and 'day_of_week' [0-6]), lacking full absolute
  timestamps (year, month, day, second) or sequencing. Furthermore, 'transaction_id' is an arbitrary identifier and
  cannot be assumed to follow chronological order. Hence, a chronological time-series split is mathematically impossible
  without introducing fabricated sequence assumptions.
- Why Customer-Grouped Evaluation is Unavailable:
  No unique customer identifier (e.g., customer_id, user_id, account_hash) exists in the schema. Group-based splitting
  or clustered evaluation across distinct customer entities is therefore impossible without inventing fictitious entities.
- Adopted Protocol:
  A rigorous stratified split across the fraud target ensures identical class proportions across all three partitions
  while preserving the exact underlying data distribution.
"""


def verify_dataset_integrity_for_splitting(df: pd.DataFrame) -> None:
    """
    Verifies that transaction_id is completely unique, target has only binary values [0, 1],
    and no missing target values exist.
    """
    if IDENTIFIER_COL not in df.columns:
        raise ValueError(f"Identifier column '{IDENTIFIER_COL}' missing from dataset.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")

    # 1. Check uniqueness of transaction_id
    if not df[IDENTIFIER_COL].is_unique:
        duplicates = df[df[IDENTIFIER_COL].duplicated(keep=False)]
        raise ValueError(f"Transaction IDs are not unique! Found {len(duplicates)} duplicate records.")

    # 2. Check target missing values
    if df[TARGET_COL].isnull().any():
        null_count = df[TARGET_COL].isnull().sum()
        raise ValueError(f"Target '{TARGET_COL}' contains {null_count} missing values.")

    # 3. Check target domain
    unique_targets = set(df[TARGET_COL].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target '{TARGET_COL}' contains invalid values: {unique_targets}. Must be {{0, 1}}.")


def create_stratified_split_manifest(
    df: Optional[pd.DataFrame] = None,
    raw_data_path: Path = RAW_DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Creates a reproducible 60/20/20 stratified split manifest linking transaction_id to split partition.
    Saves partition membership to split_manifest.csv preserving original row ordering and data.
    """
    if df is None:
        df = pd.read_csv(raw_data_path)

    verify_dataset_integrity_for_splitting(df)

    # 60% Development, 40% Temporary holdout
    dev_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df[TARGET_COL],
        random_state=random_state,
        shuffle=True,
    )

    # Split the 40% holdout equally into 20% Policy-Validation and 20% Independent Final Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[TARGET_COL],
        random_state=random_state,
        shuffle=True,
    )

    # Create manifest
    manifest_rows = []
    for tid in dev_df[IDENTIFIER_COL]:
        manifest_rows.append({IDENTIFIER_COL: tid, "split": "development"})
    for tid in val_df[IDENTIFIER_COL]:
        manifest_rows.append({IDENTIFIER_COL: tid, "split": "policy_validation"})
    for tid in test_df[IDENTIFIER_COL]:
        manifest_rows.append({IDENTIFIER_COL: tid, "split": "final_test"})

    manifest_df = pd.DataFrame(manifest_rows)
    
    # Sort manifest according to original transaction_id appearance in df
    id_to_order = {tid: idx for idx, tid in enumerate(df[IDENTIFIER_COL])}
    manifest_df["_sort_order"] = manifest_df[IDENTIFIER_COL].map(id_to_order)
    manifest_df = manifest_df.sort_values(by="_sort_order").drop(columns=["_sort_order"]).reset_index(drop=True)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)

    return manifest_df


def load_split_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Loads the split manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return create_stratified_split_manifest(manifest_path=manifest_path)
    return pd.read_csv(manifest_path)


def get_partition_data(
    partition: str,
    raw_data_path: Path = RAW_DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    expose_test_target: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """
    Safely retrieves the specified partition data with strict evaluation boundaries.
    
    Parameters:
    -----------
    partition : str
        One of 'development', 'policy_validation', 'final_test'.
    raw_data_path : Path
        Path to raw CSV dataset.
    manifest_path : Path
        Path to manifest mapping transaction_id to partition.
    expose_test_target : bool
        Security toggle. If False and partition=='final_test', target y is returned as None
        to prevent accidental evaluation leakage during development and tuning.

    Returns:
    --------
    X : pd.DataFrame
        Predictive feature matrix (23 predictors; transaction_id and fraud strictly excluded).
    y : Optional[pd.Series]
        Target series (0 or 1). Quarantined (None) for final_test unless expose_test_target=True.
    ids : pd.Series
        Transaction IDs series (strictly isolated from X).
    """
    valid_partitions = {"development", "policy_validation", "final_test"}
    if partition not in valid_partitions:
        raise ValueError(f"Invalid partition '{partition}'. Must be one of {valid_partitions}.")

    df_raw = pd.read_csv(raw_data_path)
    manifest = load_split_manifest(manifest_path)

    partition_ids = set(manifest[manifest["split"] == partition][IDENTIFIER_COL])
    part_df = df_raw[df_raw[IDENTIFIER_COL].isin(partition_ids)].copy().reset_index(drop=True)

    ids = part_df[IDENTIFIER_COL]
    
    # Strictly exclude identifier and target from feature matrix X
    feature_cols = [c for c in part_df.columns if c not in [IDENTIFIER_COL, TARGET_COL]]
    X = part_df[feature_cols].copy()

    # Target handling with quarantine guard
    if partition == "final_test" and not expose_test_target:
        y = None
    else:
        y = part_df[TARGET_COL].copy()

    return X, y, ids


def get_split_summary(
    raw_data_path: Path = RAW_DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    """
    Computes summary metrics (counts, proportions, fraud volume, fraud rates)
    across all partitions without exposing individual test labels or records.
    """
    df_raw = pd.read_csv(raw_data_path)
    manifest = load_split_manifest(manifest_path)
    merged = df_raw[[IDENTIFIER_COL, TARGET_COL]].merge(manifest, on=IDENTIFIER_COL)

    summary = merged.groupby("split")[TARGET_COL].agg(
        total_count="count",
        legit_count=lambda s: int((s == 0).sum()),
        fraud_count=lambda s: int((s == 1).sum()),
        fraud_rate=lambda s: float(s.mean()),
    ).reset_index()

    total_records = len(merged)
    summary["partition_share"] = summary["total_count"] / total_records
    summary["fraud_rate_pct"] = (summary["fraud_rate"] * 100).round(2)

    # Order partitions logically
    order_map = {"development": 0, "policy_validation": 1, "final_test": 2}
    summary["_order"] = summary["split"].map(order_map)
    summary = summary.sort_values(by="_order").drop(columns=["_order"]).reset_index(drop=True)

    return summary
