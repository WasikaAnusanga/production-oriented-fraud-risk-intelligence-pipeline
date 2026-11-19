"""
Advanced Feature Engineering Module.

Implements domain-specific fraud indicators and interaction terms via a scikit-learn
compatible transformer: AdvancedFraudFeatureEngineer.

Guarantees:
- Full compatibility with scikit-learn Pipeline (fit, transform, get_feature_names_out).
- Deterministic behavior and joblib serialization.
- Safe division by zero, missing value preservation, and negative value bounds.
- Zero data leakage: learned parameters (e.g. median transaction amount) are strictly
  estimated from the training folds during fit().
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS


class AdvancedFraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs 18 advanced fraud features
    capturing non-linear scales, velocity acceleration, spatial-device friction,
    composite risk scores, and tenure ratios.
    """

    def __init__(self, include_original: bool = True):
        self.include_original = include_original
        self.median_transaction_amount_: Optional[float] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns training-fold parameters:
        - Median transaction amount for fold-consistent amount normalization.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Estimate training-fold median transaction amount (safely handling nulls)
        if "transaction_amount" in X.columns:
            self.median_transaction_amount_ = float(X["transaction_amount"].dropna().median())
        else:
            self.median_transaction_amount_ = 1.0

        # Precompute output feature names
        dummy_transformed = self._construct_features(X.iloc[:2].copy())
        self.feature_names_out_ = list(dummy_transformed.columns)
        return self

    def _construct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal generator constructing the 18 candidate features safely.
        """
        out = pd.DataFrame(index=df.index)

        # 1. log_transaction_amount: log1p(max(amount, 0))
        if "transaction_amount" in df.columns:
            amt = df["transaction_amount"].astype(float)
            out["log_transaction_amount"] = np.where(amt.notnull(), np.log1p(np.maximum(amt, 0.0)), np.nan)
        else:
            out["log_transaction_amount"] = np.nan

        # 2. log_distance: log1p(max(distance, 0))
        if "distance_from_usual_location_km" in df.columns:
            dist = df["distance_from_usual_location_km"].astype(float)
            out["log_distance"] = np.where(dist.notnull(), np.log1p(np.maximum(dist, 0.0)), np.nan)
        else:
            out["log_distance"] = np.nan

        # 3. velocity_ratio: transactions_last_1h / (transactions_last_24h + 1)
        if "transactions_last_1h" in df.columns and "transactions_last_24h" in df.columns:
            v1 = df["transactions_last_1h"].astype(float)
            v24 = df["transactions_last_24h"].astype(float)
            out["velocity_ratio"] = v1 / (v24 + 1.0)
        else:
            out["velocity_ratio"] = np.nan

        # 4. previous_23h_transactions: max(transactions_last_24h - transactions_last_1h, 0)
        if "transactions_last_1h" in df.columns and "transactions_last_24h" in df.columns:
            v1 = df["transactions_last_1h"].astype(float)
            v24 = df["transactions_last_24h"].astype(float)
            out["previous_23h_transactions"] = np.maximum(v24 - v1, 0.0)
        else:
            out["previous_23h_transactions"] = np.nan

        # 5. amount_per_recent_transaction: transaction_amount / (transactions_last_24h + 1)
        if "transaction_amount" in df.columns and "transactions_last_24h" in df.columns:
            amt = df["transaction_amount"].astype(float)
            v24 = df["transactions_last_24h"].astype(float)
            out["amount_per_recent_transaction"] = amt / (v24 + 1.0)
        else:
            out["amount_per_recent_transaction"] = np.nan

        # 6. combined_merchant_ip_risk: merchant_risk_score * ip_risk_score
        if "merchant_risk_score" in df.columns and "ip_risk_score" in df.columns:
            m_risk = df["merchant_risk_score"].astype(float)
            ip_risk = df["ip_risk_score"].astype(float)
            out["combined_merchant_ip_risk"] = m_risk * ip_risk
        else:
            out["combined_merchant_ip_risk"] = np.nan

        # 7. merchant_ip_risk_difference: merchant_risk_score - ip_risk_score
        if "merchant_risk_score" in df.columns and "ip_risk_score" in df.columns:
            m_risk = df["merchant_risk_score"].astype(float)
            ip_risk = df["ip_risk_score"].astype(float)
            out["merchant_ip_risk_difference"] = m_risk - ip_risk
        else:
            out["merchant_ip_risk_difference"] = np.nan

        # 8. amount_merchant_risk_interaction: log1p(max(amount, 0)) * merchant_risk_score
        if "transaction_amount" in df.columns and "merchant_risk_score" in df.columns:
            amt = df["transaction_amount"].astype(float)
            m_risk = df["merchant_risk_score"].astype(float)
            log_amt = np.where(amt.notnull(), np.log1p(np.maximum(amt, 0.0)), np.nan)
            out["amount_merchant_risk_interaction"] = log_amt * m_risk
        else:
            out["amount_merchant_risk_interaction"] = np.nan

        # 9. new_device_distance_interaction: new_device * log1p(max(distance, 0))
        if "new_device" in df.columns and "distance_from_usual_location_km" in df.columns:
            nd = df["new_device"].astype(float)
            dist = df["distance_from_usual_location_km"].astype(float)
            log_d = np.where(dist.notnull(), np.log1p(np.maximum(dist, 0.0)), np.nan)
            out["new_device_distance_interaction"] = nd * log_d
        else:
            out["new_device_distance_interaction"] = np.nan

        # 10. failed_login_ip_risk: failed_logins_24h * ip_risk_score
        if "failed_logins_24h" in df.columns and "ip_risk_score" in df.columns:
            fl = df["failed_logins_24h"].astype(float)
            ip_risk = df["ip_risk_score"].astype(float)
            out["failed_login_ip_risk"] = fl * ip_risk
        else:
            out["failed_login_ip_risk"] = np.nan

        # 11. account_device_age_ratio: device_age_days / (account_age_days + 1)
        if "device_age_days" in df.columns and "account_age_days" in df.columns:
            d_age = df["device_age_days"].astype(float)
            a_age = df["account_age_days"].astype(float)
            out["account_device_age_ratio"] = d_age / (a_age + 1.0)
        else:
            out["account_device_age_ratio"] = np.nan

        # 12. email_account_age_ratio: email_age_days / (account_age_days + 1)
        if "email_age_days" in df.columns and "account_age_days" in df.columns:
            e_age = df["email_age_days"].astype(float)
            a_age = df["account_age_days"].astype(float)
            out["email_account_age_ratio"] = e_age / (a_age + 1.0)
        else:
            out["email_account_age_ratio"] = np.nan

        # 13. hour_sin: sin(2 * pi * hour / 24)
        if "hour" in df.columns:
            hr = df["hour"].astype(float)
            out["hour_sin"] = np.sin(2.0 * np.pi * hr / 24.0)
        else:
            out["hour_sin"] = np.nan

        # 14. hour_cos: cos(2 * pi * hour / 24)
        if "hour" in df.columns:
            hr = df["hour"].astype(float)
            out["hour_cos"] = np.cos(2.0 * np.pi * hr / 24.0)
        else:
            out["hour_cos"] = np.nan

        # 15. billing_distance_interaction: billing_address_mismatch * log1p(max(distance, 0))
        if "billing_address_mismatch" in df.columns and "distance_from_usual_location_km" in df.columns:
            bam = df["billing_address_mismatch"].astype(float)
            dist = df["distance_from_usual_location_km"].astype(float)
            log_d = np.where(dist.notnull(), np.log1p(np.maximum(dist, 0.0)), np.nan)
            out["billing_distance_interaction"] = bam * log_d
        else:
            out["billing_distance_interaction"] = np.nan

        # 16. chargeback_velocity_interaction: prior_chargebacks * transactions_last_24h
        if "prior_chargebacks" in df.columns and "transactions_last_24h" in df.columns:
            cb = df["prior_chargebacks"].astype(float)
            v24 = df["transactions_last_24h"].astype(float)
            out["chargeback_velocity_interaction"] = cb * v24
        else:
            out["chargeback_velocity_interaction"] = np.nan

        # 17. login_velocity_interaction: transactions_last_1h / (login_to_transaction_minutes + 1)
        if "transactions_last_1h" in df.columns and "login_to_transaction_minutes" in df.columns:
            v1 = df["transactions_last_1h"].astype(float)
            login_min = df["login_to_transaction_minutes"].astype(float)
            out["login_velocity_interaction"] = v1 / (login_min + 1.0)
        else:
            out["login_velocity_interaction"] = np.nan

        # 18. normalized_amount: transaction_amount / training_fold_median
        denom = self.median_transaction_amount_ if self.median_transaction_amount_ and self.median_transaction_amount_ > 0 else 1.0
        if "transaction_amount" in df.columns:
            amt = df["transaction_amount"].astype(float)
            out["normalized_amount"] = amt / denom
        else:
            out["normalized_amount"] = np.nan

        # Combine with original features if requested
        if self.include_original:
            return pd.concat([df.copy(), out], axis=1)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature construction to new transactions at test or production runtime.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self._construct_features(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns list of output feature column names.
        """
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        return []


def generate_feature_dictionary() -> pd.DataFrame:
    """
    Returns the comprehensive 18-feature documentation table with mathematical definitions,
    fraud-detection rationale, prediction-time availability, missing value treatments,
    and target leakage assessments.
    """
    features = [
        {
            "feature_name": "log_transaction_amount",
            "mathematical_formula": "log1p(max(transaction_amount, 0))",
            "source_columns": "transaction_amount",
            "fraud_risk_rationale": "Compresses heavy right-skewed monetary scale ($4 to $1,894), attenuating outlier leverage on linear/distance models while preserving monotonic risk ordering.",
            "prediction_time_availability": "Available immediately at transaction authorization request.",
            "missing_value_treatment": "Preserves NaN; imputed via training-fold median in downstream ColumnTransformer.",
            "leakage_risk": "None (Pre-decision monetary payload)."
        },
        {
            "feature_name": "log_distance",
            "mathematical_formula": "log1p(max(distance_from_usual_location_km, 0))",
            "source_columns": "distance_from_usual_location_km",
            "fraud_risk_rationale": "Transforms highly skewed spatial dislocation (up to 496 km). Penalizes anomalous geographical jumps non-linearly.",
            "prediction_time_availability": "Available synchronously from IP/GPS geolocation engine.",
            "missing_value_treatment": "Preserves NaN; imputed via training-fold median with indicator flag.",
            "leakage_risk": "None (Real-time geo-IP lookup)."
        },
        {
            "feature_name": "velocity_ratio",
            "mathematical_formula": "transactions_last_1h / (transactions_last_24h + 1)",
            "source_columns": "transactions_last_1h, transactions_last_24h",
            "fraud_risk_rationale": "Measures sudden burst acceleration. Ratios near 1.0 indicate high-frequency automated card-testing scripts concentrated in the past hour.",
            "prediction_time_availability": "Available from low-latency Redis sliding window counters (excluding current event).",
            "missing_value_treatment": "Source columns have 0 missing values; zero-division safe via +1 denominator.",
            "leakage_risk": "Safe under strict windowing excluding current transaction."
        },
        {
            "feature_name": "previous_23h_transactions",
            "mathematical_formula": "max(transactions_last_24h - transactions_last_1h, 0)",
            "source_columns": "transactions_last_1h, transactions_last_24h",
            "fraud_risk_rationale": "Isolates baseline daily activity preceding the most recent hour to distinguish sustained daily usage from sudden acute spikes.",
            "prediction_time_availability": "Computed from streaming window state at authorization.",
            "missing_value_treatment": "0 missing values; strictly non-negative via max(x, 0).",
            "leakage_risk": "Safe under strict streaming window definition."
        },
        {
            "feature_name": "amount_per_recent_transaction",
            "mathematical_formula": "transaction_amount / (transactions_last_24h + 1)",
            "source_columns": "transaction_amount, transactions_last_24h",
            "fraud_risk_rationale": "Calculates monetary density relative to daily frequency. High amounts on low-frequency accounts indicate account liquidation attacks.",
            "prediction_time_availability": "Available synchronously at authorization.",
            "missing_value_treatment": "Preserves NaN if amount missing; zero-division safe via +1.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "combined_merchant_ip_risk",
            "mathematical_formula": "merchant_risk_score * ip_risk_score",
            "source_columns": "merchant_risk_score, ip_risk_score",
            "fraud_risk_rationale": "Multiplicative risk composite. Joint high merchant risk and dirty IP proxy compounds risk super-linearly.",
            "prediction_time_availability": "Available from cached merchant underwriting DB and IP reputation API.",
            "missing_value_treatment": "Preserves NaN if either input is missing; downstream median imputation.",
            "leakage_risk": "Safe under point-in-time lookup SLA."
        },
        {
            "feature_name": "merchant_ip_risk_difference",
            "mathematical_formula": "merchant_risk_score - ip_risk_score",
            "source_columns": "merchant_risk_score, ip_risk_score",
            "fraud_risk_rationale": "Measures discordance between merchant reputation and network origin. High discordance highlights compromised merchant infrastructure.",
            "prediction_time_availability": "Available synchronously from upstream scoring services.",
            "missing_value_treatment": "Preserves NaN if either input missing.",
            "leakage_risk": "Safe under point-in-time lookup SLA."
        },
        {
            "feature_name": "amount_merchant_risk_interaction",
            "mathematical_formula": "log1p(max(transaction_amount, 0)) * merchant_risk_score",
            "source_columns": "transaction_amount, merchant_risk_score",
            "fraud_risk_rationale": "High dollar amounts transacted at high-risk merchant categories (Jewelry, Electronics) constitute maximum financial loss exposure.",
            "prediction_time_availability": "Available at transaction scoring request.",
            "missing_value_treatment": "Preserves NaN; imputed downstream.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "new_device_distance_interaction",
            "mathematical_formula": "new_device * log1p(max(distance_from_usual_location_km, 0))",
            "source_columns": "new_device, distance_from_usual_location_km",
            "fraud_risk_rationale": "EDA proved new devices amplify spatial risk ($23.2% vs 8.2%). Evaluates geographical displacement specifically on unrecognized hardware.",
            "prediction_time_availability": "Returned synchronously by device fingerprinting SDK and geolocation API.",
            "missing_value_treatment": "Preserves NaN if distance is missing.",
            "leakage_risk": "Safe under pre-decision SDK execution."
        },
        {
            "feature_name": "failed_login_ip_risk",
            "mathematical_formula": "failed_logins_24h * ip_risk_score",
            "source_columns": "failed_logins_24h, ip_risk_score",
            "fraud_risk_rationale": "Detects active credential brute-forcing. Failed logins from a suspicious proxy IP represent an ongoing bot takeover.",
            "prediction_time_availability": "Queried from authentication ledger and IP gateway.",
            "missing_value_treatment": "Preserves NaN if IP risk is missing.",
            "leakage_risk": "Safe under point-in-time session history."
        },
        {
            "feature_name": "account_device_age_ratio",
            "mathematical_formula": "device_age_days / (account_age_days + 1)",
            "source_columns": "device_age_days, account_age_days",
            "fraud_risk_rationale": "Captures device-account tenure divergence. Low ratio on old accounts indicates newly bound device (takeover risk); ratio > 1 indicates shared hardware.",
            "prediction_time_availability": "Computed from customer profile and device fingerprint store.",
            "missing_value_treatment": "0 missing values; zero-division safe via +1.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "email_account_age_ratio",
            "mathematical_formula": "email_age_days / (account_age_days + 1)",
            "source_columns": "email_age_days, account_age_days",
            "fraud_risk_rationale": "Evaluates digital identity consistency. Disposable emails created immediately before account registration exhibit near-zero ratios.",
            "prediction_time_availability": "Enriched via email identity provider (e.g. Emailage).",
            "missing_value_treatment": "0 missing values; zero-division safe via +1.",
            "leakage_risk": "Safe under pre-decision enrichment."
        },
        {
            "feature_name": "hour_sin",
            "mathematical_formula": "sin(2 * pi * hour / 24)",
            "source_columns": "hour",
            "fraud_risk_rationale": "Cyclic harmonic encoding mapping 23:00 and 00:00 as continuous adjacent states, preserving diurnal circularity.",
            "prediction_time_availability": "Local timestamp of transaction request.",
            "missing_value_treatment": "0 missing values in hour.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "hour_cos",
            "mathematical_formula": "cos(2 * pi * hour / 24)",
            "source_columns": "hour",
            "fraud_risk_rationale": "Orthogonal cosine cyclic component capturing daytime vs. nighttime behavioral phases.",
            "prediction_time_availability": "Local timestamp of transaction request.",
            "missing_value_treatment": "0 missing values in hour.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "billing_distance_interaction",
            "mathematical_formula": "billing_address_mismatch * log1p(max(distance_from_usual_location_km, 0))",
            "source_columns": "billing_address_mismatch, distance_from_usual_location_km",
            "fraud_risk_rationale": "Address Verification Service (AVS) mismatch combined with spatial dislocation strongly indicates stolen card credential usage.",
            "prediction_time_availability": "Returned by card network gateway during pre-authorization AVS check.",
            "missing_value_treatment": "Preserves NaN if distance is missing.",
            "leakage_risk": "Safe under pre-decision gateway response."
        },
        {
            "feature_name": "chargeback_velocity_interaction",
            "mathematical_formula": "prior_chargebacks * transactions_last_24h",
            "source_columns": "prior_chargebacks, transactions_last_24h",
            "fraud_risk_rationale": "Severe repeat offender signal. High transaction velocity on an account with historical dispute history triggers immediate manual review.",
            "prediction_time_availability": "Internal historical chargeback ledger.",
            "missing_value_treatment": "0 missing values.",
            "leakage_risk": "Safe under point-in-time dispute records."
        },
        {
            "feature_name": "login_velocity_interaction",
            "mathematical_formula": "transactions_last_1h / (login_to_transaction_minutes + 1)",
            "source_columns": "transactions_last_1h, login_to_transaction_minutes",
            "fraud_risk_rationale": "Measures transaction frequency per minute of session duration. High velocity immediately following login indicates automated API script execution.",
            "prediction_time_availability": "Session management token timestamps.",
            "missing_value_treatment": "Preserves NaN if login minutes missing; zero-division safe via +1.",
            "leakage_risk": "None."
        },
        {
            "feature_name": "normalized_amount",
            "mathematical_formula": "transaction_amount / training_fold_median_amount",
            "source_columns": "transaction_amount",
            "fraud_risk_rationale": "Measures dollar size relative to typical training-fold baseline ($51.78). Scaling factor learned strictly inside training folds.",
            "prediction_time_availability": "Immediate authorization payload.",
            "missing_value_treatment": "Preserves NaN; imputed downstream.",
            "leakage_risk": "Zero leakage: median parameter estimated strictly from training fold."
        }
    ]
    return pd.DataFrame(features)


# Backward-compatible alias
FraudFeatureEngineer = AdvancedFraudFeatureEngineer
