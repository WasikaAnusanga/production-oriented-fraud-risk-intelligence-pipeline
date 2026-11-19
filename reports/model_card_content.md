# Model Card: Fraud Risk Intelligence Calibrated Decision Engine
**Model Identifier**: `fraud-risk-intelligence-v2-calibrated-rf`  
**Model Version**: `1.0.0`  
**Model Date**: 2026-11-19  
**Artifact Path**: `04_calibrated_model.joblib` (Production Calibrated Pipeline), `03_fraud_model_pipeline.joblib` (Base Estimator Pipeline)  
**Developers**: Senior ML & MLOps Engineering Team  
**License**: Proprietary / Enterprise Payment Security

---

## 1. Intended Use & Deployment Scope
- **Primary Objective**: Real-time payment transaction risk assessment and 3-tier operational triage for retail card-not-present (CNP) and electronic commerce transactions.
- **Decision Engine Actions**:
  - **`APPROVE` ($p < 0.10$)**: Low-risk transactions processed seamlessly without user friction.
  - **`MANUAL REVIEW` ($0.10 \le p < 0.36$)**: Medium-risk transactions routed to human fraud investigator queues for step-up verification.
  - **`BLOCK` ($p \ge 0.36$)**: High-risk transactions declined automatically to prevent immediate financial loss.
- **Intended Users**: Retail bank fraud operations teams, e-commerce payment risk analysts, and MLOps monitoring engineers.

## 2. Prohibited Uses & Regulatory Boundaries
- **Prohibited Domains**: Credit underwriting, credit limit determination, tenant screening, employment background evaluation, or insurance pricing.
- **Fair Lending Compliance**: The model must never be used to make adverse credit decisions governed by the Equal Credit Opportunity Act (ECOA) or Fair Credit Reporting Act (FCRA).
- **Protected Characteristics**: The model does not include age, gender, race, ethnicity, or postal code demographic proxies.

## 3. Input Schema & Feature Pipeline
The pipeline accepts raw 23-predictor transaction DataFrames directly. The `transaction_id` identifier and `fraud` target are strictly excluded from predictive features.

### Core Predictor Attributes:
- **Numerical (16)**: `hour`, `day_of_week`, `transaction_amount`, `distance_from_usual_location_km`, `transactions_last_1h`, `transactions_last_24h`, `account_age_days`, `failed_logins_24h`, `prior_chargebacks`, `device_age_days`, `merchant_risk_score`, `ip_risk_score`, `email_age_days`, `login_to_transaction_minutes`, `new_device`, `billing_address_mismatch`.
- **Categorical (7)**: `card_presence`, `channel`, `country`, `device_type`, `merchant_category`, `authentication_level`, `customer_tier`.

### Internal Preprocessing & Feature Engineering:
1. **Engineered Predictors (18 features)**: Velocity ratios, logarithmic monetary scaling (`log1p`), compound merchant-IP risk interactions, hardware-distance interactions, and login velocity indicators.
2. **Missingness Imputation**: Median imputation with explicit binary missingness indicator columns for numerical predictors. Informative categorical missingness (`authentication_level`) is preserved as a distinct `'Missing'` category.
3. **Scaling & Encoding**: `StandardScaler` for numeric predictors; `OneHotEncoder(handle_unknown='ignore')` yielding 85 transformed columns.

## 4. Model Architecture & Training Methodology
- **Base Architecture**: `RandomForestClassifier` (200 trees, max depth 8, min samples split 10, min samples leaf 4, `max_features='sqrt'`).
- **Probability Calibration**: 5-fold cross-fitted **Sigmoid (Platt) Scaling** via `CalibratedClassifierCV(cv=5)` fitted exclusively on the 1,800 development partition records.
- **Training Partition Governance**: 60% Development ($n=1,800$), 20% Policy-Validation ($n=600$), 20% Independent Final Test ($n=600$). Zero leakage into final test.

## 5. Performance Metrics & Validation Benchmarks

### A. Development Cross-Validation Performance (5-Fold CV, n=1,800):
- **PR-AUC (Average Precision)**: **$0.2335 \pm 0.0242$** (Significantly superior to random baseline $12.11\%$).
- **ROC-AUC**: **$0.6627 \pm 0.0437$**
- **Brier Score**: **$0.1027 \pm 0.0018$**
- **Log Loss**: **$0.3531 \pm 0.0065$**
- **Expected Calibration Error (ECE)**: **$0.0178$**

### B. Independent Final Holdout Test Evaluation (n=600, Untouched):
- **ROC-AUC**: **$0.6402$**
- **PR-AUC**: **$0.2003$**
- **Brier Score**: **$0.1041$**
- **Expected Calibration Error (ECE)**: **$0.0108$**
- **Balanced Accuracy**: **$59.79\%$**
- **Recall (Triage Flag at $p \ge 0.10$)**: **$69.86\%$** ($51$ of $73$ test frauds captured).
- **Precision (Triage Flag)**: **$16.14\%$**
- **Financial Cost per 1,000 Transactions**: **$11,100.01\text{ CU}$** (**$42.10\%$ cost reduction** vs. default binary baseline of $19,171.57\text{ CU}$).
- **False Positive Hard Blocks**: **$0.57\%$** (only $3$ legitimate customers incorrectly blocked).
- **Manual Review Workload**: **$52.00\%$** ($312$ transactions).

## 6. Explainability & Regulatory Reason Codes
- **Global Importance (TreeSHAP)**: Top predictive drivers are `login_velocity_interaction`, `combined_merchant_ip_risk`, `login_to_transaction_minutes`, `amount_merchant_risk_interaction`, and `log_transaction_amount`.
- **Adverse Action Attribution**: For all reviewed and blocked transactions, the engine derives top-3 risk increasing and risk decreasing factors using TreeSHAP margin attributions.
- **Attribution Boundary**: SHAP values decompose the base Random Forest margins (log-odds), which are mapped to posterior probabilities via Platt sigmoid scaling.

## 7. Operational Limitations & Known Failure Modes
1. **Promotional Velocity Vulnerability**: Rapid transaction acceleration (`transactions_last_1h`) elevates fraud scores significantly. Stress testing revealed that simulated velocity surges increase review workload to $81.33\%$ and cost by $+51.58\%$.
2. **Missing Full Timestamps**: The dataset provides only cyclical `hour` and `day_of_week`. Seasonal holiday trends and long-term macroeconomic shifts cannot be captured chronologically.
3. **Capacity Sensitivity**: Achieving $69.86\%$ fraud capture requires routing $52.00\%$ of transactions to review. When review bandwidth is restricted, the adaptive operating threshold ($t_{\text{review}}=0.18$) must be utilized.

## 8. Continuous Monitoring, Drift Triggers & Retraining Policy
- **Daily Score Drift**: Alert MLOps if daily score distribution Wasserstein distance exceeds $\mathcal{W}_1 > 0.05$.
- **Weekly Feature Stability**: Audit data feeds if Population Stability Index (PSI) $> 0.20$ on top predictors.
- **Telemetry Influx Canary**: Raise Sev-2 alert if missingness in `authentication_level` exceeds $25\%$.
- **Retraining Cycle**: Quarterly scheduled retraining, or triggered immediately if 60-day confirmed fraud recall drops below $60.0\%$ or Brier score degrades $> 0.120$.
- **Emergency Kill Switch**: In the event of upstream risk API failure, automated BLOCK is disabled ($t_{\text{block}}=1.0$) to prevent customer declinations, diverting traffic to manual review and 3DS step-up authentication.
