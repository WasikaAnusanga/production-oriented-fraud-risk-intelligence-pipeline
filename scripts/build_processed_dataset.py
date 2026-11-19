"""
Generation Script for 02_processed_dataset.csv.

Applies the frozen feature engineering and preprocessing transformations fitted EXCLUSIVELY
on the Development partition (1,800 records) to the complete dataset, guaranteeing:
- ZERO train-test leakage (imputation medians, standard scalers, and encoders fitted on dev only).
- Retains transaction_id for auditing.
- Clearly distinguishes target column 'fraud' and 'split' partition membership.
- Exports 02_processed_dataset.csv in the project root.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from config.settings import (
    RAW_DATA_PATH,
    SPLITS_DIR,
    IDENTIFIER_COL,
    TARGET_COL,
    NUMERIC_PREDICTORS,
    CATEGORICAL_PREDICTORS,
)
from src.data_splitting import load_split_manifest
from src.feature_engineering import AdvancedFraudFeatureEngineer
from src.preprocessing import build_preprocessor, ALL_NUMERIC_FEATURES
from src.explainability import get_pipeline_feature_names


def build_processed_dataset():
    print("Building 02_processed_dataset.csv with strict zero train-test leakage...")

    # 1. Load raw dataset and split manifest
    raw_df = pd.read_csv(RAW_DATA_PATH)
    manifest = load_split_manifest()
    merged_df = raw_df.merge(manifest, on=IDENTIFIER_COL, how="left")

    # 2. Extract Development partition for fitting transformers
    dev_mask = merged_df["split"] == "development"
    dev_df = merged_df[dev_mask].copy()

    X_dev = dev_df[NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS]

    # 3. Fit Feature Engineering and ColumnTransformer EXCLUSIVELY on Development data
    fe = AdvancedFraudFeatureEngineer(include_original=True)
    X_dev_fe = fe.fit_transform(X_dev)

    preprocessor = build_preprocessor(
        numeric_features=ALL_NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_PREDICTORS,
        scale_numeric=True,
    )
    preprocessor.fit(X_dev_fe)

    # 4. Transform the Entire Dataset (Development, Policy-Validation, Final-Test)
    X_all = merged_df[NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS]
    X_all_fe = fe.transform(X_all)
    X_all_proc = preprocessor.transform(X_all_fe)
    if hasattr(X_all_proc, "toarray"):
        X_all_proc = X_all_proc.toarray()

    # Get feature names
    feature_names = [f"feat_{name}" for name in preprocessor.get_feature_names_out()]

    # 5. Assemble Processed DataFrame
    processed_df = pd.DataFrame(X_all_proc, columns=feature_names)

    # Insert auditing identifiers, split membership, and target
    processed_df.insert(0, IDENTIFIER_COL, merged_df[IDENTIFIER_COL])
    processed_df.insert(1, "split_partition", merged_df["split"])
    processed_df.insert(2, TARGET_COL, merged_df[TARGET_COL])

    # 6. Save to 02_processed_dataset.csv in project root
    out_path = Path("02_processed_dataset.csv")
    processed_df.to_csv(out_path, index=False)
    print(f"Successfully created {out_path} ({len(processed_df)} rows, {processed_df.shape[1]} columns)")


if __name__ == "__main__":
    build_processed_dataset()
