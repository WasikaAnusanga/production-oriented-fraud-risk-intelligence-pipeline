# Fraud Risk Intelligence Pipeline (`fraud-risk-intelligence-v2`)


[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Project Overview
The **Fraud Risk Intelligence Pipeline** is an end-to-end, production-oriented machine learning and operational decisioning system designed to detect payment fraud in retail e-commerce and card-not-present transactions.

Rather than relying on naive accuracy or arbitrary $0.50$ classification thresholds, this pipeline implements:
1. **Calibrated Posterior Probabilities**: Utilizing cross-fitted Platt/Sigmoid scaling to output statistically valid empirical fraud likelihoods.
2. **3-Tier Operational Triage Decision Engine**: Routing transactions into `APPROVE`, `MANUAL REVIEW`, and `BLOCK` based on asymmetric financial loss matrices.
3. **Explainable AI (XAI)**: Generating TreeSHAP global feature attributions and local adverse action regulatory reason codes.
4. **Production MLOps Governance**: Comprehensive stress testing, drift bounds, schema checksums, and monitoring metric specifications.

---

## 2. Directory Structure
```text
fraud-risk-intelligence/
├── 01_fraud_risk_analysis.ipynb          # Principal Executed Analysis Notebook (Sections A-K)
├── 02_processed_dataset.csv              # Full dataset with leakage-free engineered features (3,000 rows)
├── 03_fraud_model_pipeline.joblib        # Fitted base estimator pipeline (RF + preprocessing)
├── 04_calibrated_model.joblib            # Production calibrated pipeline (accepts raw DataFrames)
├── 05_threshold_cost_analysis.csv        # Complete 1,176-point policy threshold grid evaluation
├── 08_README.md                          # Comprehensive system documentation
├── README.md                             # Repository entry documentation
├── pyproject.toml                        # Project configuration and build specifications
├── setup.py                              # Package installation script
├── requirements.txt                      # Environment dependencies
├── config/
│   ├── settings.py                       # Global settings, paths, seeds, and column schemas
│   ├── business_costs.yaml               # Cost units (CU) and business scenario definitions
│   └── frozen_pipeline_config.yaml       # Frozen production model and threshold parameters
├── data/
│   ├── raw/                              # Immutable raw benchmark dataset (3,000 rows, 25 cols)
│   └── splits/                           # Split manifest (60% Dev, 20% Val, 20% Test)
├── src/                                  # Modular production source package
│   ├── data_audit.py                     # Data quality forensics and missingness tests (Section A)
│   ├── data_splitting.py                 # Stratified partitioning and leakage quarantine (Stage 2)
│   ├── eda.py                            # Exploratory data analysis and bivariate visualizations (Section B)
│   ├── feature_engineering.py            # Custom scikit-learn transformer (18 features) (Section C)
│   ├── preprocessing.py                  # ColumnTransformer, imputation, scaling, OHE
│   ├── model_training.py                 # Estimator pipelines, CV evaluation, tuning (Sections D & G)
│   ├── calibration.py                    # CalibratedClassifierCV, ECE, reliability diagrams (Section F)
│   ├── cost_optimization.py              # Financial loss functions, grid search, sensitivity (Section E)
│   ├── explainability.py                 # TreeSHAP, beeswarm, dependence, local waterfalls (Section H)
│   ├── fraud_decision_engine.py          # Production decision engine & operations simulation (Section I)
│   ├── robustness.py                     # Stress testing scenarios 1-8 & fold stability (Section J)
│   └── governance.py                     # Checksums, monitoring metrics, lifecycle policies (Section K)
├── tests/                                # 49 automated pytest unit and integration tests
├── scripts/                              # Standalone reproducibility execution scripts
└── reports/
    ├── report_evidence.json              # Complete verified evidentiary data package
    ├── report_findings.md                # Comprehensive technical summary & discussion answers
    ├── model_card_content.md             # Formal model card documentation
    ├── figures/                          # 25 publication-quality 300 DPI visualizations
    └── tables/                           # 26 statistical and operational benchmark CSV tables
```

---

## 3. Environment Setup & Installation

### Prerequisites
- Python 3.10 to 3.14
- Virtual environment tool (`venv`)

### Installation Commands
```powershell
# 1. Clone repository and navigate to root
cd fraud-risk-intelligence

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install package in editable mode (without build isolation)
pip install --no-build-isolation -e . --no-deps
```

---

## 4. Execution & Pipeline Reproducibility

### Running Automated Test Suite
To execute the complete 49-test suite verifying all components:
```powershell
pytest tests/ -v
```
*Current test status: **49 passed in 23.72s**.*

### Rebuilding Artifacts from Scratch
Individual stages can be executed via dedicated scripts:
```powershell
# 1. Generate Stratified Splits
python -c "from src.data_splitting import create_stratified_split_manifest; create_stratified_split_manifest()"

# 2. Run Data Forensics (Section A)
python -c "from src.data_audit import run_full_data_audit; run_full_data_audit()"

# 3. Run EDA Figures (Section B)
python -c "from src.eda import run_eda_pipeline; run_eda_pipeline()"

# 4. Train and Benchmark Models (Sections D & G)
python scripts/train_and_evaluate_models.py

# 5. Evaluate Probability Calibration (Section F)
python scripts/evaluate_calibration.py

# 6. Run Cost-Sensitive Threshold Optimization (Section E)
python scripts/evaluate_cost_optimization.py

# 7. Generate Explainability Attributions (Section H)
python scripts/evaluate_explainability.py

# 8. Run Fraud Operations Simulation (Section I)
python scripts/evaluate_operations_simulation.py

# 9. Execute Robustness Stress Tests (Section J)
python scripts/evaluate_robustness.py

# 10. Generate Processed Dataset
python scripts/build_processed_dataset.py

# 11. Evaluate Independent Final Holdout (Section K)
python scripts/evaluate_final_test.py

# 12. Compile Evidence Package
python scripts/compile_report_evidence.py
```

### Executing the Principal Jupyter Notebook
The principal deliverable `01_fraud_risk_analysis.ipynb` contains the fully executed, end-to-end reproducible analysis across Sections A through K.
```powershell
jupyter nbconvert --to notebook --execute --inplace 01_fraud_risk_analysis.ipynb
```

---

## 5. Production Model Inference

The production calibrated pipeline accepts raw 23-predictor transaction DataFrames directly without requiring manual feature engineering, scaling, or imputation:

```python
import joblib
import pandas as pd
from src.fraud_decision_engine import FraudDecisionEngine

# 1. Initialize Decision Engine
engine = FraudDecisionEngine(
    model_path="04_calibrated_model.joblib",
    review_threshold=0.10,
    block_threshold=0.36
)

# 2. Load Raw Transaction Records (target 'fraud' is NEVER required)
raw_txns = pd.DataFrame([{
    "transaction_id": "TXN_LIVE_001",
    "hour": 22,
    "day_of_week": 4,
    "transaction_amount": 340.50,
    "distance_from_usual_location_km": 125.0,
    "transactions_last_1h": 4,
    "transactions_last_24h": 9,
    "account_age_days": 45,
    "failed_logins_24h": 2,
    "prior_chargebacks": 1,
    "device_age_days": 12,
    "merchant_risk_score": 0.82,
    "ip_risk_score": 0.74,
    "email_age_days": 30,
    "login_to_transaction_minutes": 0.8,
    "new_device": 1,
    "billing_address_mismatch": 1,
    "card_presence": "Not Present",
    "channel": "E-commerce",
    "country": "US",
    "device_type": "Mobile",
    "merchant_category": "Electronics",
    "authentication_level": "None",
    "customer_tier": "Standard"
}])

# 3. Predict Probability, Operational Decision, and Reason Codes
results = engine.predict_transaction_risk(raw_txns, include_explanations=True)
print(results[["transaction_id", "fraud_probability", "operational_decision", "risk_level", "plain_language_reason"]])
```

---

## 6. Key Empirical Results Summary

| Metric | Development (5-Fold CV) | Policy-Validation ($n=600$) | Final Independent Test ($n=600$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | $0.6627 \pm 0.0437$ | $0.6363$ | **$0.6402$** |
| **PR-AUC** | $0.2335 \pm 0.0242$ | $0.1917$ | **$0.2003$** |
| **Brier Score** | $0.1027 \pm 0.0018$ | $0.1030$ | **$0.1041$** |
| **Expected Calibration Error** | $0.0178$ | $0.0178$ | **$0.0108$** |
| **Fraud Capture Rate** | - | $73.97\%$ | **$69.86\%$** ($51$ of $73$ frauds caught) |
| **Manual Review Workload** | - | $52.83\%$ | **$52.00\%$** ($312$ transactions) |
| **Legitimate Hard Blocks** | - | $3$ ($0.57\%$) | **$3$ ($0.57\%$)** |
| **Cost per 1k Txns (CU)** | - | $9,201.08\text{ CU}$ | **$11,100.01\text{ CU}$** (**$42.10\%$ savings**) |

---

## 7. Known Limitations & Governance Constraints
1. **Timestamp Granularity**: Cyclical `hour` and `day_of_week` are available; full chronological timestamps are absent. Time-based chronological validation requires ISO-8601 logging in future deployments.
2. **Review Capacity Dependency**: The cost-optimal policy routes $52.00\%$ of volume to manual review. For capacity-constrained operations, an adaptive review threshold ($t_{\text{review}}=0.18$) is documented.
3. **Velocity Drift Vulnerability**: Input velocity spikes (`transactions_last_1h`) significantly increase model risk scores. Production deployment requires rate-limiting filters and promotional velocity throttling.
