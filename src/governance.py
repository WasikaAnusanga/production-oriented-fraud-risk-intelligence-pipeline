"""
Model Governance and Production Integrity Module.

Defines the enterprise governance policy, auditing protocols, schema checksums,
and monitoring specifications for the Fraud Risk Intelligence Pipeline:
- Intended & prohibited model use definitions.
- Prediction-time feature availability & leakage audits.
- Production monitoring specifications across 8 formal metrics.
- Drift detection, alert triggers, threshold review, retraining, and rollback criteria.
- Operational kill switch (safe automated block disabling).
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import hashlib
import json
import pandas as pd
import numpy as np

from config.settings import ALL_PREDICTORS, RAW_DATA_PATH


def compute_dataset_checksum(filepath: str = str(RAW_DATA_PATH)) -> str:
    """Computes SHA-256 checksum to guarantee raw data immutability."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_production_governance_policy() -> Dict[str, Any]:
    """Returns the comprehensive model governance policy dictionary."""
    return {
        "intended_use": (
            "Automated payment risk triage for retail e-commerce and card-not-present transactions. "
            "Routes transactions into APPROVE (p < 0.10), MANUAL REVIEW (0.10 <= p < 0.36), and BLOCK (p >= 0.36)."
        ),
        "prohibited_use": (
            "Credit scoring, creditworthiness evaluations, automated employment background screening, "
            "tenant vetting, or legal jurisdiction profiling. Disallowed under ECOA and FCRA."
        ),
        "dataset_assumptions": (
            "Cross-sectional transaction snapshot with 12.13% historical fraud prevalence. "
            "Predictors represent telemetry available strictly at authorization time."
        ),
        "prediction_time_availability": (
            "Strict isolation: transaction_id and fraud target are quarantined. All 23 predictors "
            "must be populated from authorization stream telemetry."
        ),
        "target_leakage_mitigation": (
            "Zero post-transaction settlement or chargeback feedback variables are permitted into the feature matrix. "
            "Prior chargebacks represent historical merchant records, not current transaction outcomes."
        ),
        "temporal_limitations": (
            "Only hour and day_of_week are provided; full chronological timestamps are unavailable. "
            "Stratified random partitioning is utilized; chronological time-based drift models require future timestamp logging."
        ),
        "monitoring_frequency": "Daily score distributions; weekly drift sweeps; monthly retraining reviews.",
    }


def get_monitoring_metric_specifications() -> List[Dict[str, Any]]:
    """
    Defines 8 formal production monitoring metrics with operational triggers and response actions.
    """
    return [
        {
            "metric_name": "Prediction Score Distribution Drift",
            "measurement_definition": "Wasserstein distance (Earth Mover's Distance) between daily score distribution and baseline policy validation distribution.",
            "required_data": "Daily inference log of calibrated fraud probabilities.",
            "monitoring_frequency": "Daily",
            "proposed_trigger": "Wasserstein distance > 0.05 or median probability shifts by > 0.03.",
            "recommended_response": "Trigger MLOps investigation; verify upstream merchant telemetry; inspect feature distributions.",
        },
        {
            "metric_name": "Important Feature Distribution Drift",
            "measurement_definition": "Population Stability Index (PSI) on top 5 global predictors (e.g. combined_merchant_ip_risk, login_velocity_interaction).",
            "required_data": "Raw transaction feature telemetry from authorization stream.",
            "monitoring_frequency": "Weekly",
            "proposed_trigger": "PSI > 0.20 on any top-5 feature (significant distributional shift).",
            "recommended_response": "Audit feature pipelines; verify upstream risk score vendor feeds; check for holiday/promotional velocity surges.",
        },
        {
            "metric_name": "Missing-Value Influx Rate",
            "measurement_definition": "Percentage of missing values in core telemetry fields (authentication_level, merchant_risk_score, ip_risk_score).",
            "required_data": "Daily transaction feature payload logs.",
            "monitoring_frequency": "Daily / Hourly Canary",
            "proposed_trigger": "Missingness > 25% in authentication_level or > 10% in risk score feeds.",
            "recommended_response": "Raise Sev-2 telemetry degradation alert; inspect mobile app/web SDK network timeouts; verify API vendor uptime.",
        },
        {
            "metric_name": "Confirmed Fraud Recall (Delayed Feedback)",
            "measurement_definition": "True positive capture rate based on 30-to-60 day confirmed chargeback and manual review clearance records.",
            "required_data": "Historical authorization logs joined with settled dispute records.",
            "monitoring_frequency": "Monthly (with rolling 60-day lag)",
            "proposed_trigger": "Realized fraud capture rate drops below 60.0% (baseline: 69.86%).",
            "recommended_response": "Initiate immediate model retraining cycle; re-optimize operational decision thresholds; review evasion patterns.",
        },
        {
            "metric_name": "False-Positive Customer Insult Burden",
            "measurement_definition": "Rate of legitimate transactions incorrectly hard-blocked relative to total volume.",
            "required_data": "Customer support dispute logs, chargeback clearance logs, and automated block logs.",
            "monitoring_frequency": "Weekly",
            "proposed_trigger": "False hard block rate exceeds 1.5% of legitimate volume (baseline: 0.57%).",
            "recommended_response": "Elevate block_threshold (e.g. from 0.36 to 0.42); route borderline cases to manual review; audit top fraud rules.",
        },
        {
            "metric_name": "Probability Calibration Reliability (Brier & ECE)",
            "measurement_definition": "Empirical Brier score and Expected Calibration Error (ECE) on confirmed historical transaction cohorts.",
            "required_data": "Monthly settled transaction outcomes and logged calibrated probabilities.",
            "monitoring_frequency": "Monthly",
            "proposed_trigger": "Brier score > 0.120 (baseline: 0.104) or ECE > 0.040 (baseline: 0.011).",
            "recommended_response": "Refit CalibratedClassifierCV calibrator on recent 90-day settled development window; verify Platt scaling parameters.",
        },
        {
            "metric_name": "Manual Review Queue Workload",
            "measurement_definition": "Percentage of total transaction volume routed to human investigator queues.",
            "required_data": "Real-time decision engine transaction routing logs.",
            "monitoring_frequency": "Daily / Shift-by-Shift",
            "proposed_trigger": "Manual review demand exceeds 65.0% of total volume or investigator queue latency exceeds SLA (e.g. > 15 minutes).",
            "recommended_response": "Engage adaptive capacity throttling (raise review_threshold from 0.10 to 0.15); route lowest-risk review band to passive step-up auth.",
        },
        {
            "metric_name": "Expected Business Cost per 1,000 Transactions",
            "measurement_definition": "Realized financial cost units (CU) per 1k transactions combining fraud loss, review wages, and customer friction.",
            "required_data": "Monthly financial operational reporting ledger.",
            "monitoring_frequency": "Monthly",
            "proposed_trigger": "Cost per 1k transactions exceeds 14,000 CU (baseline: 11,100 CU; a >25% inflation shock).",
            "recommended_response": "Executive review of cost parameters; investigate sudden large-amount fraud leakage; recalibrate 3-tier policy.",
        },
    ]


def get_operational_lifecycle_actions() -> Dict[str, Any]:
    """Specifies governance conditions for drift, threshold review, retraining, rollback, and kill switch."""
    return {
        "investigate_drift": (
            "Triggered when daily Wasserstein score drift exceeds 0.05 or weekly feature PSI exceeds 0.20. "
            "MLOps reviews data feeds without halting inference."
        ),
        "review_thresholds": (
            "Triggered when review queue congestion exceeds 65% capacity or confirmed false block rate exceeds 1.5%. "
            "Risk committee adjusts review_threshold or block_threshold within approved bounds."
        ),
        "retrain_model": (
            "Triggered quarterly as standard maintenance, or ad-hoc if 60-day confirmed fraud recall drops below 60.0% "
            "or Brier score exceeds 0.120. Retraining must follow frozen pipeline CV protocols."
        ),
        "rollback_to_prior_version": (
            "Triggered if new model deployment exhibits unexpected runtime error rate > 0.1%, severe calibration degradation (ECE > 0.05), "
            "or severe unexplained drop in fraud capture within first 48 hours."
        ),
        "temporarily_disable_automated_blocking": (
            "OPERATIONAL KILL SWITCH: If incoming telemetry experiences systemic upstream failure (e.g. merchant risk score API "
            "down, missingness > 50%), automated BLOCK is immediately suspended (block_threshold set to 1.0). "
            "All suspect transactions are diverted to MANUAL REVIEW or step-up 3DS authentication to prevent widespread customer lockouts."
        ),
    }
