"""
Data Forensics and Quality Audit Module.

Provides comprehensive auditing, structural profiling, data hygiene checks,
missingness pattern analysis, extreme value detection, and leakage assessment.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

from config.settings import (
    RAW_DATA_PATH,
    IDENTIFIER_COL,
    TARGET_COL,
    NUMERIC_PREDICTORS,
    CATEGORICAL_PREDICTORS,
    ALL_COLUMNS,
    EXPECTED_ROW_COUNT,
    EXPECTED_COL_COUNT,
    EXPECTED_FRAUD_COUNT,
    EXPECTED_LEGIT_COUNT,
    EXPECTED_MISSING_COUNTS,
)


def load_raw_dataset(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Safely loads the raw transactions dataset.
    Raises FileNotFoundError if file is missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found at path: {filepath.resolve()}")
    return pd.read_csv(filepath)


def verify_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifies column count, column names, identifier, target, and predictor groupings.
    """
    actual_cols = list(df.columns)
    missing_cols = set(ALL_COLUMNS) - set(actual_cols)
    extra_cols = set(actual_cols) - set(ALL_COLUMNS)

    is_valid = (
        len(actual_cols) == EXPECTED_COL_COUNT
        and len(missing_cols) == 0
        and len(extra_cols) == 0
        and IDENTIFIER_COL in actual_cols
        and TARGET_COL in actual_cols
    )

    return {
        "is_valid": is_valid,
        "actual_col_count": len(actual_cols),
        "expected_col_count": EXPECTED_COL_COUNT,
        "missing_columns": list(missing_cols),
        "unexpected_columns": list(extra_cols),
        "numeric_predictors_present": [c for c in NUMERIC_PREDICTORS if c in actual_cols],
        "categorical_predictors_present": [c for c in CATEGORICAL_PREDICTORS if c in actual_cols],
    }


def verify_dimensions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifies that the row and column dimensions match verified benchmark.
    """
    n_rows, n_cols = df.shape
    return {
        "is_valid": (n_rows == EXPECTED_ROW_COUNT and n_cols == EXPECTED_COL_COUNT),
        "actual_rows": n_rows,
        "expected_rows": EXPECTED_ROW_COUNT,
        "actual_cols": n_cols,
        "expected_cols": EXPECTED_COL_COUNT,
    }


def verify_target_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifies target counts: 0 = legitimate (2,636), 1 = fraudulent (364).
    """
    counts = df[TARGET_COL].value_counts().to_dict()
    actual_legit = counts.get(0, 0)
    actual_fraud = counts.get(1, 0)
    fraud_rate = actual_fraud / len(df) if len(df) > 0 else 0.0

    is_valid = (
        actual_legit == EXPECTED_LEGIT_COUNT
        and actual_fraud == EXPECTED_FRAUD_COUNT
    )

    return {
        "is_valid": is_valid,
        "legit_count": actual_legit,
        "expected_legit": EXPECTED_LEGIT_COUNT,
        "fraud_count": actual_fraud,
        "expected_fraud": EXPECTED_FRAUD_COUNT,
        "fraud_rate": fraud_rate,
        "fraud_rate_pct": round(fraud_rate * 100, 2),
    }


def verify_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits null values and verifies them against the known missing values specification.
    """
    null_counts = df.isnull().sum()
    actual_missing = null_counts[null_counts > 0].to_dict()

    matches_spec = True
    discrepancies = {}

    for col, expected_count in EXPECTED_MISSING_COUNTS.items():
        actual_count = actual_missing.get(col, 0)
        if actual_count != expected_count:
            matches_spec = False
            discrepancies[col] = {
                "expected": expected_count,
                "actual": actual_count,
            }

    for col, actual_count in actual_missing.items():
        if col not in EXPECTED_MISSING_COUNTS:
            matches_spec = False
            discrepancies[col] = {
                "expected": 0,
                "actual": actual_count,
            }

    return {
        "is_valid": matches_spec,
        "actual_missing": actual_missing,
        "expected_missing": EXPECTED_MISSING_COUNTS,
        "discrepancies": discrepancies,
    }


def run_full_dataset_audit(filepath: Path = RAW_DATA_PATH) -> Dict[str, Any]:
    """
    Orchestrates full dataset integrity audit.
    """
    df = load_raw_dataset(filepath)
    dim_audit = verify_dimensions(df)
    schema_audit = verify_schema(df)
    target_audit = verify_target_distribution(df)
    missing_audit = verify_missing_values(df)

    overall_pass = (
        dim_audit["is_valid"]
        and schema_audit["is_valid"]
        and target_audit["is_valid"]
        and missing_audit["is_valid"]
    )

    return {
        "overall_pass": overall_pass,
        "dimensions": dim_audit,
        "schema": schema_audit,
        "target": target_audit,
        "missing_values": missing_audit,
    }


# ==============================================================================
# STAGE 3: COMPREHENSIVE DATA FORENSICS & QUALITY ENGINEERING (DEVELOPMENT SET)
# ==============================================================================

def analyze_dataset_structure(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Profiles all columns: types, cardinalities, missingness, numeric percentiles,
    skewness, and categorical distributions.
    """
    # 1. Column Overview Table
    overview = []
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        n_missing = s.isnull().sum()
        pct_missing = (n_missing / n_rows) * 100
        n_unique = s.nunique(dropna=True)
        dtype = str(s.dtype)
        col_type = (
            "Identifier" if col == IDENTIFIER_COL
            else "Target" if col == TARGET_COL
            else "Categorical" if col in CATEGORICAL_PREDICTORS
            else "Numeric"
        )
        overview.append({
            "column_name": col,
            "column_type": col_type,
            "dtype": dtype,
            "non_null_count": n_rows - n_missing,
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "cardinality": n_unique,
            "sample_values": str(s.dropna().unique()[:3].tolist())
        })
    overview_df = pd.DataFrame(overview)

    # 2. Numeric Detailed Summary (percentiles, skewness, IQR)
    num_cols = [c for c in NUMERIC_PREDICTORS if c in df.columns]
    num_summary = []
    for col in num_cols:
        s = df[col].dropna()
        q25 = s.quantile(0.25)
        q50 = s.quantile(0.50)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        num_summary.append({
            "feature": col,
            "count": len(s),
            "missing_count": df[col].isnull().sum(),
            "mean": round(s.mean(), 3),
            "std": round(s.std(), 3),
            "min": round(s.min(), 3),
            "p01": round(s.quantile(0.01), 3),
            "p05": round(s.quantile(0.05), 3),
            "p25 (Q1)": round(q25, 3),
            "p50 (Median)": round(q50, 3),
            "p75 (Q3)": round(q75, 3),
            "p95": round(s.quantile(0.95), 3),
            "p99": round(s.quantile(0.99), 3),
            "max": round(s.max(), 3),
            "iqr": round(iqr, 3),
            "skewness": round(s.skew(), 3)
        })
    num_summary_df = pd.DataFrame(num_summary)

    # 3. Categorical Frequencies & Rare Levels (< 5% prevalence)
    cat_cols = [c for c in CATEGORICAL_PREDICTORS if c in df.columns]
    cat_summary = []
    for col in cat_cols:
        val_counts = df[col].value_counts(dropna=False)
        total = len(df)
        for val, count in val_counts.items():
            pct = (count / total) * 100
            val_str = "MISSING (NaN)" if pd.isna(val) else str(val)
            cat_summary.append({
                "feature": col,
                "category": val_str,
                "count": count,
                "percentage": round(pct, 2),
                "is_rare (<5%)": (pct < 5.0) and not pd.isna(val)
            })
    cat_summary_df = pd.DataFrame(cat_summary)

    return overview_df, num_summary_df, cat_summary_df


def run_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes domain validation checks on development transactions:
    - Duplicates (exact rows, transaction_id)
    - Value boundary violations (hour [0-23], day_of_week [0-6], risk scores [0-1])
    - Negative constraints (amounts, distances, counts, ages)
    - Binary indicator domain violations
    - Relational logical violations (transactions_last_1h > transactions_last_24h)
    - Non-integer count values
    - Suspicious combinations (e.g. login_to_transaction_minutes == 0, physical card in e-commerce)
    """
    checks = []

    # 1. Duplicate transaction_id
    dup_ids = df[IDENTIFIER_COL].duplicated().sum() if IDENTIFIER_COL in df.columns else 0
    checks.append({
        "check_category": "Uniqueness",
        "rule_description": "Unique transaction_id",
        "violation_count": int(dup_ids),
        "status": "PASS" if dup_ids == 0 else "FAIL",
        "details": f"{dup_ids} duplicate transaction IDs found."
    })

    # 2. Exact duplicate rows
    dup_rows = df.duplicated().sum()
    checks.append({
        "check_category": "Uniqueness",
        "rule_description": "Zero exact duplicate rows across all features",
        "violation_count": int(dup_rows),
        "status": "PASS" if dup_rows == 0 else "FAIL",
        "details": f"{dup_rows} exact duplicate rows identified."
    })

    # 3. Hour range [0, 23]
    if "hour" in df.columns:
        invalid_hours = (~df["hour"].dropna().between(0, 23)).sum()
        checks.append({
            "check_category": "Range Boundary",
            "rule_description": "Hour strictly in range [0, 23]",
            "violation_count": int(invalid_hours),
            "status": "PASS" if invalid_hours == 0 else "FAIL",
            "details": f"{invalid_hours} hour values out of bounds."
        })

    # 4. Day of week range [0, 6]
    if "day_of_week" in df.columns:
        invalid_dow = (~df["day_of_week"].dropna().between(0, 6)).sum()
        checks.append({
            "check_category": "Range Boundary",
            "rule_description": "Day of week strictly in range [0, 6]",
            "violation_count": int(invalid_dow),
            "status": "PASS" if invalid_dow == 0 else "FAIL",
            "details": f"{invalid_dow} day_of_week values out of bounds."
        })

    # 5. Non-negative checks
    non_negative_cols = [
        "transaction_amount", "distance_from_usual_location_km",
        "transactions_last_1h", "transactions_last_24h", "account_age_days",
        "failed_logins_24h", "prior_chargebacks", "device_age_days",
        "email_age_days", "login_to_transaction_minutes"
    ]
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = (df[col].dropna() < 0).sum()
            checks.append({
                "check_category": "Non-Negativity",
                "rule_description": f"{col} must be non-negative (>= 0)",
                "violation_count": int(neg_count),
                "status": "PASS" if neg_count == 0 else "FAIL",
                "details": f"{neg_count} negative entries found in {col}."
            })

    # 6. Risk scores in [0.0, 1.0]
    for score_col in ["merchant_risk_score", "ip_risk_score"]:
        if score_col in df.columns:
            out_of_bounds = (~df[score_col].dropna().between(0.0, 1.0)).sum()
            checks.append({
                "check_category": "Range Boundary",
                "rule_description": f"{score_col} must be bounded in [0.0, 1.0]",
                "violation_count": int(out_of_bounds),
                "status": "PASS" if out_of_bounds == 0 else "FAIL",
                "details": f"{out_of_bounds} values outside [0.0, 1.0] for {score_col}."
            })

    # 7. Binary indicators {0, 1}
    binary_cols = ["new_device", "billing_address_mismatch"]
    if TARGET_COL in df.columns:
        binary_cols.append(TARGET_COL)
    for b_col in binary_cols:
        if b_col in df.columns:
            invalid_binary = (~df[b_col].dropna().isin([0, 1])).sum()
            checks.append({
                "check_category": "Binary Domain",
                "rule_description": f"{b_col} must only contain values in {{0, 1}}",
                "violation_count": int(invalid_binary),
                "status": "PASS" if invalid_binary == 0 else "FAIL",
                "details": f"{invalid_binary} non-binary values in {b_col}."
            })

    # 8. Relational consistency: transactions_last_1h <= transactions_last_24h
    if "transactions_last_1h" in df.columns and "transactions_last_24h" in df.columns:
        velocity_violations = (df["transactions_last_1h"] > df["transactions_last_24h"]).sum()
        checks.append({
            "check_category": "Relational Logic",
            "rule_description": "transactions_last_1h <= transactions_last_24h",
            "violation_count": int(velocity_violations),
            "status": "PASS" if velocity_violations == 0 else "FAIL",
            "details": f"{velocity_violations} records where 1h velocity exceeds 24h velocity."
        })

    # 9. Integer check on count features
    count_cols = [
        "transactions_last_1h", "transactions_last_24h", "failed_logins_24h",
        "prior_chargebacks", "account_age_days", "device_age_days", "email_age_days"
    ]
    for c_col in count_cols:
        if c_col in df.columns:
            s_drop = df[c_col].dropna()
            non_int = (s_drop % 1 != 0).sum()
            checks.append({
                "check_category": "Type Integrity",
                "rule_description": f"{c_col} must be integer count values",
                "violation_count": int(non_int),
                "status": "PASS" if non_int == 0 else "FAIL",
                "details": f"{non_int} non-integer values detected in {c_col}."
            })

    # 10. Suspicious contextual combinations
    # Physical card present in E-commerce channel (rare/suspicious but possible in specialized setups)
    if "card_presence" in df.columns and "channel" in df.columns:
        card_ecom = ((df["card_presence"] == "Present") & (df["channel"] == "E-commerce")).sum()
        checks.append({
            "check_category": "Domain Context",
            "rule_description": "Flag card_presence='Present' with channel='E-commerce'",
            "violation_count": int(card_ecom),
            "status": "FLAG_SUSPICIOUS" if card_ecom > 0 else "PASS",
            "details": f"{card_ecom} records exhibit physical card present in e-commerce transactions."
        })

    # 0 login minutes with failed logins > 0 (immediate bot transaction)
    if "login_to_transaction_minutes" in df.columns and "failed_logins_24h" in df.columns:
        bot_pattern = ((df["login_to_transaction_minutes"] == 0) & (df["failed_logins_24h"] > 0)).sum()
        checks.append({
            "check_category": "Domain Context",
            "rule_description": "Flag instant login (0 min) combined with failed_logins_24h > 0",
            "violation_count": int(bot_pattern),
            "status": "FLAG_SUSPICIOUS" if bot_pattern > 0 else "PASS",
            "details": f"{bot_pattern} transactions executed within 0 minutes of login with prior failed attempts."
        })

    return pd.DataFrame(checks)


def analyze_missingness_patterns(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Performs rigorous missingness analysis on the six features with missing values.
    Calculates missing count, missing %, observed vs missing fraud rate,
    Fisher exact or Chi-Square contingency statistical test, p-value, and effect size (Odds Ratio / Cramer's V).
    Operates strictly on the development partition.
    """
    missing_cols = [
        "authentication_level",
        "distance_from_usual_location_km",
        "merchant_risk_score",
        "login_to_transaction_minutes",
        "transaction_amount",
        "ip_risk_score",
    ]

    records = []
    n_dev = len(df)
    base_fraud_rate = df[target_col].mean()

    for col in missing_cols:
        if col not in df.columns:
            continue

        is_missing = df[col].isnull()
        missing_count = int(is_missing.sum())
        missing_pct = round((missing_count / n_dev) * 100, 2)
        observed_count = n_dev - missing_count

        # Contingency table: [is_missing, is_fraud]
        # [[obs_legit, obs_fraud], [miss_legit, miss_fraud]]
        miss_fraud = int(df.loc[is_missing, target_col].sum())
        miss_legit = missing_count - miss_fraud
        obs_fraud = int(df.loc[~is_missing, target_col].sum())
        obs_legit = observed_count - obs_fraud

        fraud_rate_missing = round((miss_fraud / missing_count) * 100, 2) if missing_count > 0 else 0.0
        fraud_rate_observed = round((obs_fraud / observed_count) * 100, 2) if observed_count > 0 else 0.0

        contingency_table = np.array([
            [obs_legit, obs_fraud],
            [miss_legit, miss_fraud]
        ])

        # Fisher's exact test (exact for small sample cells) and Chi-square
        odds_ratio, p_val_fisher = stats.fisher_exact(contingency_table)
        chi2, p_val_chi2, _, _ = stats.chi2_contingency(contingency_table, correction=True)

        # Cramer's V as effect size
        total_n = n_dev
        cramers_v = np.sqrt(chi2 / total_n) if total_n > 0 else 0.0

        # Substantive interpretation
        is_stat_sig = p_val_fisher < 0.05
        diff_pct = round(fraud_rate_missing - fraud_rate_observed, 2)
        if is_stat_sig:
            direction = "elevated" if diff_pct > 0 else "suppressed"
            interpretation = (
                f"Statistically significant association (p={p_val_fisher:.4f}, OR={odds_ratio:.2f}). "
                f"Missingness is predictive (MNAR/MAR candidate); fraud rate is {direction} by {abs(diff_pct):.2f}%."
            )
        else:
            interpretation = (
                f"No statistically significant difference (p={p_val_fisher:.4f}, OR={odds_ratio:.2f}). "
                f"Missingness appears non-informative (MCAR candidate)."
            )

        records.append({
            "feature": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "observed_count": observed_count,
            "fraud_count_missing": miss_fraud,
            "fraud_rate_missing_pct": fraud_rate_missing,
            "fraud_count_observed": obs_fraud,
            "fraud_rate_observed_pct": fraud_rate_observed,
            "fraud_rate_diff_pct": diff_pct,
            "p_value_fisher": round(p_val_fisher, 5),
            "odds_ratio": round(odds_ratio, 3),
            "cramers_v": round(cramers_v, 4),
            "is_significant_05": is_stat_sig,
            "predictive_missingness_interpretation": interpretation
        })

    return pd.DataFrame(records)


def detect_extreme_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Investigates extreme observations and heavy tails using Tukey's IQR rule
    and extreme percentiles (p01, p99, p99.9). Flags outliers without automatic deletion.
    """
    investigate_cols = [
        "transaction_amount",
        "distance_from_usual_location_km",
        "transactions_last_1h",
        "transactions_last_24h",
        "merchant_risk_score",
        "ip_risk_score",
        "login_to_transaction_minutes"
    ]

    records = []
    n_dev = len(df)

    for col in investigate_cols:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        upper_whisker = q75 + 1.5 * iqr
        lower_whisker = max(0.0, q25 - 1.5 * iqr)
        extreme_upper = q75 + 3.0 * iqr

        p95 = s.quantile(0.95)
        p99 = s.quantile(0.99)
        max_val = s.max()

        mild_outliers = ((s > upper_whisker) | (s < lower_whisker)).sum()
        extreme_outliers = (s > extreme_upper).sum()
        pct_outliers = (mild_outliers / len(s)) * 100

        # Fraud rate among extreme outliers vs non-outliers
        is_upper_outlier = df[col] > upper_whisker
        outlier_fraud_rate = df.loc[is_upper_outlier, TARGET_COL].mean() if is_upper_outlier.sum() > 0 else np.nan
        normal_fraud_rate = df.loc[~is_upper_outlier, TARGET_COL].mean() if (~is_upper_outlier).sum() > 0 else np.nan

        records.append({
            "feature": col,
            "non_null_n": len(s),
            "q25": round(q25, 2),
            "median": round(s.median(), 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "upper_fence_1_5_iqr": round(upper_whisker, 2),
            "extreme_fence_3_0_iqr": round(extreme_upper, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max_val, 2),
            "mild_outlier_count": int(mild_outliers),
            "extreme_outlier_count": int(extreme_outliers),
            "outlier_pct": round(pct_outliers, 2),
            "fraud_rate_in_outliers_pct": round(outlier_fraud_rate * 100, 2) if not np.isnan(outlier_fraud_rate) else np.nan,
            "fraud_rate_in_inliers_pct": round(normal_fraud_rate * 100, 2) if not np.isnan(normal_fraud_rate) else np.nan,
            "recommendation": "Flag & robust-scale / log1p transform (do NOT remove: high fraud concentration in extremes)."
        })

    return pd.DataFrame(records)


def assess_target_leakage() -> pd.DataFrame:
    """
    Evaluates potential target leakage for all critical fields.
    Documents whether each value is genuinely available at runtime decisioning,
    and establishes operational eligibility.
    """
    evaluations = [
        {
            "feature": "fraud",
            "runtime_available": "No (Target Outcome)",
            "leakage_risk": "Direct Target Leakage",
            "decision": "EXCLUDED",
            "operational_justification": "Ground truth label; knowing the label before decisioning is circular. Strictly excluded from inputs."
        },
        {
            "feature": "transaction_id",
            "runtime_available": "Yes (Metadata)",
            "leakage_risk": "Spurious Non-Predictive Identifier",
            "decision": "EXCLUDED",
            "operational_justification": "Arbitrary unique identifier string. Using it leads to memorization and severe test overfitting."
        },
        {
            "feature": "prior_chargebacks",
            "runtime_available": "Yes (Pre-transaction Historical Ledger)",
            "leakage_risk": "Safe under Point-in-Time Ledger Assumption",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Available at transaction authorization if retrieved from real-time customer ledger. Must strictly record chargebacks prior to current transaction."
        },
        {
            "feature": "merchant_risk_score",
            "runtime_available": "Yes (Merchant Profile Service)",
            "leakage_risk": "Safe under Offline/Real-time Lookups",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Computed periodically by merchant underwriting or third-party consortium. Assumed cached at authorization time."
        },
        {
            "feature": "ip_risk_score",
            "runtime_available": "Yes (IP Geolocation/Reputation API)",
            "leakage_risk": "Safe under Third-Party Enrichment SLA",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Returned synchronously by IP reputation gateway (e.g. MaxMind) during checkout session."
        },
        {
            "feature": "authentication_level",
            "runtime_available": "Yes (3DS / Protocol Handshake)",
            "leakage_risk": "Safe (Pre-decision Step)",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Represents 3D-Secure or multi-factor authentication protocol negotiated prior to scoring."
        },
        {
            "feature": "login_to_transaction_minutes",
            "runtime_available": "Yes (Session Management Service)",
            "leakage_risk": "Safe (Session Metadata)",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Difference between transaction timestamp and customer session login timestamp."
        },
        {
            "feature": "transactions_last_1h",
            "runtime_available": "Yes (Streaming Feature Store / Redis)",
            "leakage_risk": "Safe under Strict Real-time Windowing",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Sliding-window counter maintained in caching layer. Must strictly exclude current transaction."
        },
        {
            "feature": "transactions_last_24h",
            "runtime_available": "Yes (Streaming Feature Store / Redis)",
            "leakage_risk": "Safe under Strict Real-time Windowing",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Sliding-window counter maintained in caching layer. Must strictly exclude current transaction."
        },
        {
            "feature": "device_age_days",
            "runtime_available": "Yes (Device Fingerprinting SDK)",
            "leakage_risk": "Safe (Client Fingerprint DB)",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Calculated by client-side fraud SDK as elapsed days since device fingerprint first recorded."
        },
        {
            "feature": "new_device",
            "runtime_available": "Yes (Device Fingerprinting SDK)",
            "leakage_risk": "Safe (Client Fingerprint DB)",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Binary indicator emitted by device fingerprinting service if device GUID has no prior history."
        },
        {
            "feature": "billing_address_mismatch",
            "runtime_available": "Yes (AVS - Address Verification Service)",
            "leakage_risk": "Safe (Pre-authorization Gateway Result)",
            "decision": "CONDITIONALLY USABLE",
            "operational_justification": "Returned by card payment gateway during AVS check prior to fraud decisioning engine."
        }
    ]
    return pd.DataFrame(evaluations)


def build_data_treatment_decisions() -> pd.DataFrame:
    """
    Constructs an explicit data treatment decision table across all features and defect types.
    Precludes global fitting and documents inside-pipeline transformer specifications.
    """
    decisions = [
        {
            "feature_or_issue": "authentication_level (Categorical Missingness: 289 nulls)",
            "observed_defect": "16.0% missing; missingness shows significant association with fraud.",
            "treatment_decision": "Categorical Missing Imputation ('MISSING') + One-Hot Encoding",
            "pipeline_mechanism": "SimpleImputer(strategy='constant', fill_value='MISSING') inside ColumnTransformer.",
            "justification": "Preserves distinct predictive missingness signal without imputing arbitrary mode."
        },
        {
            "feature_or_issue": "distance_from_usual_location_km (Numeric Missingness: 75 nulls)",
            "observed_defect": "4.2% missing; heavy right skew (max=496 km).",
            "treatment_decision": "Median Imputation + Missing Indicator Flag",
            "pipeline_mechanism": "SimpleImputer(strategy='median', add_indicator=True) inside ColumnTransformer.",
            "justification": "Median is robust to heavy spatial outliers; missing indicator captures geo-sensor absence."
        },
        {
            "feature_or_issue": "merchant_risk_score (Numeric Missingness: 60 nulls)",
            "observed_defect": "3.3% missing; bounded [0, 1].",
            "treatment_decision": "Median Imputation + Missing Indicator Flag",
            "pipeline_mechanism": "SimpleImputer(strategy='median', add_indicator=True) inside ColumnTransformer.",
            "justification": "Provides neutral risk proxy for unrated merchants while preserving unrated indicator."
        },
        {
            "feature_or_issue": "login_to_transaction_minutes (Numeric Missingness: 60 nulls)",
            "observed_defect": "3.3% missing; right-skewed session duration.",
            "treatment_decision": "Median Imputation + Missing Indicator Flag",
            "pipeline_mechanism": "SimpleImputer(strategy='median', add_indicator=True) inside ColumnTransformer.",
            "justification": "Preserves neutral session duration while flagging headless/direct API sessions."
        },
        {
            "feature_or_issue": "transaction_amount (Numeric Missingness: 53 nulls)",
            "observed_defect": "2.9% missing; financial monetary predictor.",
            "treatment_decision": "Median Imputation + Missing Indicator Flag",
            "pipeline_mechanism": "SimpleImputer(strategy='median', add_indicator=True) inside ColumnTransformer.",
            "justification": "Median prevents distortion from large transactions; indicator captures missing price signal."
        },
        {
            "feature_or_issue": "ip_risk_score (Numeric Missingness: 45 nulls)",
            "observed_defect": "2.5% missing; external reputation score.",
            "treatment_decision": "Median Imputation + Missing Indicator Flag",
            "pipeline_mechanism": "SimpleImputer(strategy='median', add_indicator=True) inside ColumnTransformer.",
            "justification": "Robust imputation for external lookup failures."
        },
        {
            "feature_or_issue": "transaction_amount & distance (Extreme Outliers)",
            "observed_defect": "Heavy right tails; extreme values carry 2-3x higher fraud rate.",
            "treatment_decision": "Retain All Extremes + Robust Feature Scaling / Non-linear Tree Models",
            "pipeline_mechanism": "StandardScaler / RobustScaler fitted within CV folds; tree algorithms (XGBoost/LightGBM).",
            "justification": "Never delete valid high-value fraud transactions; models must learn high-amount risk."
        },
        {
            "feature_or_issue": "Categorical Features (channel, country, device, etc.)",
            "observed_defect": "Multi-category nominal variables with low-cardinality rare levels.",
            "treatment_decision": "One-Hot Encoding with handle_unknown='ignore'",
            "pipeline_mechanism": "OneHotEncoder(handle_unknown='ignore', sparse_output=False).",
            "justification": "Prevents unseen category errors at test/production inference time."
        },
        {
            "feature_or_issue": "Identifier & Target (transaction_id, fraud)",
            "observed_defect": "Potential leakage and spurious memorization.",
            "treatment_decision": "Strict Exclusion from Feature Matrix X",
            "pipeline_mechanism": "Separated prior to ColumnTransformer via get_partition_data loader.",
            "justification": "Guarantees zero target leakage and zero non-predictive identifier overfitting."
        }
    ]
    return pd.DataFrame(decisions)
