"""
Tests for Reproducible Data Splitting and Evaluation Governance (Stage 2).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config.settings import (
    RAW_DATA_PATH,
    SPLITS_DIR,
    IDENTIFIER_COL,
    TARGET_COL,
    RANDOM_STATE,
    EXPECTED_ROW_COUNT,
    EXPECTED_COL_COUNT,
    ALL_PREDICTORS,
)
from src.data_splitting import (
    MANIFEST_PATH,
    create_stratified_split_manifest,
    load_split_manifest,
    get_partition_data,
    get_split_summary,
    verify_dataset_integrity_for_splitting,
)


def test_dataset_integrity_check():
    """Verify integrity assertion rejects duplicated ids or corrupted targets."""
    df_raw = pd.read_csv(RAW_DATA_PATH)
    verify_dataset_integrity_for_splitting(df_raw)

    # Corrupt ID test
    df_dup = df_raw.copy()
    df_dup.loc[1, IDENTIFIER_COL] = df_dup.loc[0, IDENTIFIER_COL]
    with pytest.raises(ValueError, match="not unique"):
        verify_dataset_integrity_for_splitting(df_dup)

    # Corrupt target test (NaN)
    df_nan = df_raw.copy()
    df_nan.loc[0, TARGET_COL] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        verify_dataset_integrity_for_splitting(df_nan)

    # Corrupt target test (out of domain)
    df_bad = df_raw.copy()
    df_bad.loc[0, TARGET_COL] = 2
    with pytest.raises(ValueError, match="invalid values"):
        verify_dataset_integrity_for_splitting(df_bad)


def test_split_manifest_existence_and_dimensions():
    """Verify split manifest file exists and matches original dataset rows."""
    manifest = load_split_manifest()
    assert MANIFEST_PATH.exists()
    assert len(manifest) == EXPECTED_ROW_COUNT
    assert set(manifest.columns) == {IDENTIFIER_COL, "split"}


def test_partition_sizes_and_proportions():
    """Verify 60% Development (1800), 20% Policy-Validation (600), 20% Final Test (600)."""
    manifest = load_split_manifest()
    counts = manifest["split"].value_counts().to_dict()

    assert counts["development"] == 1800
    assert counts["policy_validation"] == 600
    assert counts["final_test"] == 600


def test_nonoverlapping_partitions():
    """Verify zero overlap in transaction_id across all three partitions."""
    manifest = load_split_manifest()
    dev_ids = set(manifest[manifest["split"] == "development"][IDENTIFIER_COL])
    val_ids = set(manifest[manifest["split"] == "policy_validation"][IDENTIFIER_COL])
    test_ids = set(manifest[manifest["split"] == "final_test"][IDENTIFIER_COL])

    assert len(dev_ids.intersection(val_ids)) == 0, "Overlap between Dev and Policy-Val"
    assert len(dev_ids.intersection(test_ids)) == 0, "Overlap between Dev and Final Test"
    assert len(val_ids.intersection(test_ids)) == 0, "Overlap between Policy-Val and Final Test"
    assert len(dev_ids | val_ids | test_ids) == EXPECTED_ROW_COUNT


def test_stratification_and_class_proportions():
    """Verify class proportions match across splits (~12.1% fraud rate)."""
    summary = get_split_summary()
    assert len(summary) == 3

    for _, row in summary.iterrows():
        # Overall fraud rate is 364/3000 = 12.133%
        # Proportions should be tightly bounded around ~12.1%
        assert 12.0 <= row["fraud_rate_pct"] <= 12.3, f"Unexpected fraud rate in {row['split']}: {row['fraud_rate_pct']}"


def test_split_reproducibility():
    """Verify that re-running manifest creation with random_state=42 yields identical results."""
    m1 = load_split_manifest()
    m2 = create_stratified_split_manifest(random_state=RANDOM_STATE)
    pd.testing.assert_frame_equal(m1, m2)


def test_feature_matrix_isolation_and_quarantine():
    """
    Verify transaction_id and fraud never enter X.
    Verify final_test target is quarantined unless explicitly requested.
    """
    # 1. Development partition
    X_dev, y_dev, ids_dev = get_partition_data("development")
    assert IDENTIFIER_COL not in X_dev.columns
    assert TARGET_COL not in X_dev.columns
    assert X_dev.shape == (1800, 23)
    assert len(y_dev) == 1800
    assert len(ids_dev) == 1800

    # 2. Policy-validation partition
    X_val, y_val, ids_val = get_partition_data("policy_validation")
    assert IDENTIFIER_COL not in X_val.columns
    assert TARGET_COL not in X_val.columns
    assert X_val.shape == (600, 23)
    assert len(y_val) == 600

    # 3. Final test partition (Quarantine check)
    X_test, y_test_quarantined, ids_test = get_partition_data("final_test", expose_test_target=False)
    assert IDENTIFIER_COL not in X_test.columns
    assert TARGET_COL not in X_test.columns
    assert X_test.shape == (600, 23)
    assert y_test_quarantined is None, "Final test target was not quarantined!"

    # 4. Final test partition (Explicit un-quarantine)
    _, y_test_exposed, _ = get_partition_data("final_test", expose_test_target=True)
    assert y_test_exposed is not None
    assert len(y_test_exposed) == 600
    assert set(y_test_exposed.unique()) == {0, 1}
