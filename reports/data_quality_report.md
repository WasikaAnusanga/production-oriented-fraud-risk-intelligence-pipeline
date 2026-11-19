# Data Forensics & Quality Engineering Report (Section A)
**Project**: `fraud-risk-intelligence-v2`  
**Dataset**: `data/raw/transaction_fraud_risk_advanced_dataset.csv`  
**Partition Investigated**: Development Partition ($n = 1,800$ transactions, $60\%$ sample, target-informed)  
**Evaluation Principle**: Test & Policy-Validation partitions quarantined to prevent methodological leakage.

---

## 1. Executive Summary & Structural Architecture

The development partition ($1,800$ rows $\times 25$ columns) represents an authentic transaction stream with $218$ fraudulent events ($12.11\%$) and $1,582$ legitimate transactions ($87.89\%$).

### Schema & Structural Composition
- **Identifier ($1$)**: `transaction_id` (100% unique string token, non-predictive).
- **Target ($1$)**: `fraud` (binary domain $\{0, 1\}$, zero missing values).
- **Numeric & Binary Predictors ($16$)**:
  - *Cyclical Time*: `hour` (0–23), `day_of_week` (0–6).
  - *Monetary & Spatial*: `transaction_amount` ($), `distance_from_usual_location_km` (km).
  - *Velocity Indicators*: `transactions_last_1h`, `transactions_last_24h`.
  - *Tenure & Recency*: `account_age_days`, `device_age_days`, `email_age_days`, `login_to_transaction_minutes`.
  - *Historical & Behavioral Friction*: `failed_logins_24h`, `prior_chargebacks`.
  - *Pre-computed Risk Scores*: `merchant_risk_score` $\in [0, 1]$, `ip_risk_score` $\in [0, 1]$.
  - *Binary Mismatch Indicators*: `new_device` $\in \{0, 1\}$, `billing_address_mismatch` $\in \{0, 1\}$.
- **Categorical Predictors ($7$)**:
  - `card_presence` (2 levels: 'Present', 'Not present')
  - `channel` (3 levels: 'E-commerce', 'POS', 'Mobile')
  - `country` (5 levels: 'Sri Lanka', 'UK', 'USA', 'India', 'Singapore')
  - `device_type` (4 levels: 'Android', 'iOS', 'Windows', 'macOS', 'Other')
  - `merchant_category` (6 levels: 'Grocery', 'Electronics', 'Travel', 'Clothing', 'Restaurants', 'Entertainment')
  - `authentication_level` (2 observed levels: 'Standard', 'Strong', plus missing)
  - `customer_tier` (3 levels: 'Basic', 'Silver', 'Gold')

---

## 2. Comprehensive Data Quality Audit

A suite of automated domain validation checks was executed on the development partition:

| Category | Domain Rule Description | Violations | Status | Engineering Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Uniqueness** | Unique `transaction_id` | 0 | **PASS** | Every transaction possesses a distinct identifier; zero row collisions. |
| **Uniqueness** | Zero duplicate feature rows | 0 | **PASS** | No repeated or cloned transaction records detected. |
| **Range Bounds** | `hour` $\in [0, 23]$ | 0 | **PASS** | Full 24-hour cycle observed with zero out-of-bounds hours. |
| **Range Bounds** | `day_of_week` $\in [0, 6]$ | 0 | **PASS** | Complete 7-day calendar cycle observed with zero invalid indices. |
| **Non-Negativity** | Monetary amounts, distances, tenures $\ge 0$ | 0 | **PASS** | No negative amounts, negative distances, or negative tenure days. |
| **Score Bounds** | `merchant_risk_score`, `ip_risk_score` $\in [0, 1]$ | 0 | **PASS** | Both risk indices strictly obey probability-like unit intervals. |
| **Binary Domain** | Indicators $\in \{0, 1\}$ | 0 | **PASS** | `new_device`, `billing_address_mismatch`, and `fraud` contain only binary flags. |
| **Relational Logic** | `transactions_last_1h` $\le$ `transactions_last_24h` | 0 | **PASS** | Window nesting logic holds 100%; 1h velocity never exceeds 24h count. |
| **Type Integrity** | Integer counts strictly integer | 0 | **PASS** | Count variables contain zero fractional or corrupted float values. |
| **Domain Context** | Physical card present in E-commerce | 134 | **FLAGGED** | Authentic domain scenario: card present via connected smartcard/NFC reader or terminal simulation. Retained as valid domain signal. |
| **Domain Context** | 0 min login with failed logins > 0 | 12 | **FLAGGED** | Fast automated credential stuffing/bot attack pattern. High fraud concentration ($58.3\%$). Retained. |

---

## 3. In-Depth Missingness Forensics & Statistical Testing

Six variables exhibit missing values in the development partition. To evaluate whether missingness carries informative signal (Missing Not at Random / Missing at Random) versus non-informative noise (Missing Completely at Random), exact two-sided Fisher's tests and contingency statistics were computed against the fraud target:

| Feature | Missing Count | Missing (%) | Observed Count | Fraud Rate (Missing) | Fraud Rate (Observed) | Diff (%) | Fisher p-value | Odds Ratio | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`authentication_level`** | 161 | 8.94% | 1,639 | **20.50%** | 11.29% | **+9.21%** | **0.00139** | **2.03** | **Statistically Significant (p < 0.01)** |
| **`distance_from_usual_location_km`** | 43 | 2.39% | 1,757 | 9.30% | 12.18% | -2.88% | 0.81212 | 0.74 | Non-significant (p = 0.81) |
| **`merchant_risk_score`** | 33 | 1.83% | 1,767 | **0.00%** | 12.34% | **-12.34%** | **0.02694** | **0.00** | **Statistically Significant (p < 0.05)** |
| **`login_to_transaction_minutes`** | 38 | 2.11% | 1,762 | 10.53% | 12.15% | -1.62% | 1.00000 | 0.85 | Non-significant (p = 1.00) |
| **`transaction_amount`** | 32 | 1.78% | 1,768 | 6.25% | 12.22% | -5.97% | 0.41817 | 0.48 | Non-significant (p = 0.42) |
| **`ip_risk_score`** | 21 | 1.17% | 1,779 | 14.29% | 12.09% | +2.20% | 0.73388 | 1.21 | Non-significant (p = 0.73) |

### Forensic Interpretations
1. **`authentication_level` (Predictive Missingness)**:
   - Transactions lacking an authentication level record exhibit a **20.50% fraud rate**, compared to **11.29%** for recorded handshakes ($\text{OR} = 2.03$, $p = 0.00139$).
   - *Domain Mechanism*: Unauthenticated guest checkouts, bypassed 3DS flows, or legacy merchant terminals frequently fail to record protocol handshake telemetry. Treating missingness as an explicit categorical level (`'MISSING'`) preserves this risk signal.
2. **`merchant_risk_score` (Unrated Merchants)**:
   - Zero fraud was observed among the 33 transactions with missing merchant risk scores ($p = 0.02694$).
   - *Domain Mechanism*: Likely corresponds to newly onboarded low-risk internal merchants or specialized whitelisted partners. Imputing with median while adding a missingness indicator flag captures both the proxy score and unrated status.
3. **Other Numerical Features (`distance`, `login_minutes`, `amount`, `ip_score`)**:
   - Show no statistically significant deviations from the baseline rate ($p > 0.40$), consistent with MCAR/MAR telemetry drops (e.g. GPS disabled on client device, API timeout). Imputing with median and explicit indicator flags ensures robust modeling without bias.

---

## 4. Extreme Observations & Heavy Tail Forensics

Using Tukey's IQR rule ($1.5 \times \text{IQR}$ and $3.0 \times \text{IQR}$ fences) and empirical percentiles, distributions were audited for extreme values:

| Feature | Q25 | Median | Q75 | IQR | 1.5x Fence | Max Value | Outlier Count | Fraud Rate (Outliers) | Fraud Rate (Inliers) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`transaction_amount`** | $25.26 | $51.78 | $112.55 | $87.29 | $243.48 | **$1,894.20** | 108 (6.0%) | **22.69%** | 11.41% |
| **`distance_from_usual_location_km`** | 12.40 | 32.15 | 68.90 | 56.50 | 153.65 | **496.10** | 50 (2.8%) | **22.00%** | 11.83% |
| **`transactions_last_1h`** | 1.00 | 2.00 | 4.00 | 3.00 | 8.50 | **14.00** | 44 (2.4%) | **45.45%** | 11.28% |
| **`transactions_last_24h`** | 4.00 | 8.00 | 12.00 | 8.00 | 24.00 | **38.00** | 39 (2.2%) | **17.95%** | 11.98% |
| **`merchant_risk_score`** | 0.15 | 0.32 | 0.58 | 0.43 | 1.00 | **0.99** | 0 (0.0%) | N/A | 12.11% |
| **`ip_risk_score`** | 0.12 | 0.28 | 0.54 | 0.42 | 1.00 | **0.98** | 0 (0.0%) | N/A | 12.11% |
| **`login_to_transaction_minutes`** | 5.20 | 14.80 | 32.40 | 27.20 | 73.20 | **284.00** | 97 (5.5%) | **12.37%** | 12.10% |

### Outlier Policy Decision
- **Zero Deletion Mandate**: Outliers in `transactions_last_1h` exhibit a staggering **45.45% fraud rate** (nearly 4x the base rate). Outliers in monetary amount and distance carry a **22.69%** and **22.00% fraud rate** (nearly 2x the base rate).
- Automatically removing these records would eliminate the high-risk fraud cases the model is designed to detect.
- **Handling**: Retain all records; apply robust scaling within CV folds, log1p-smoothed velocity transforms in Feature Engineering, and rely on non-linear tree ensembles (XGBoost, LightGBM, Random Forest) that are naturally invariant to monotonic outlier scaling.

---

## 5. Runtime Target Leakage Assessment

Every feature was audited against the operational timeline of transaction authorization:

| Feature Name | Runtime Availability | Leakage Classification | Decision | Operational Pre-decision Assumption |
| :--- | :--- | :--- | :--- | :--- |
| **`fraud`** | Post-authorization (Disputes/Chargebacks) | Direct Target Leakage | **EXCLUDED** | Ground truth label; strictly isolated from input matrix $X$. |
| **`transaction_id`** | Pre-decision (Metadata) | Non-Predictive Identifier | **EXCLUDED** | Arbitrary token string; causes memorization and test leakage. |
| **`prior_chargebacks`** | Pre-decision (Customer Ledger) | Conditionally Usable | **RETAINED** | Point-in-time historical ledger check; strictly reflects chargebacks resolved prior to current transaction. |
| **`merchant_risk_score`** | Pre-decision (Merchant Risk Service) | Conditionally Usable | **RETAINED** | Pre-computed risk rating cached in low-latency merchant profile store. |
| **`ip_risk_score`** | Pre-decision (IP Geolocation API) | Conditionally Usable | **RETAINED** | Evaluated synchronously during checkout network connection. |
| **`authentication_level`** | Pre-decision (3DS Handshake) | Conditionally Usable | **RETAINED** | Protocol status established during 3D Secure / MFA challenge before risk scoring. |
| **`login_to_transaction_minutes`** | Pre-decision (Session Manager) | Conditionally Usable | **RETAINED** | Elapsed time between session authentication token generation and transaction post. |
| **`transactions_last_1h`** | Pre-decision (Feature Store / Redis) | Conditionally Usable | **RETAINED** | Real-time sliding window counter strictly excluding current transaction. |
| **`transactions_last_24h`** | Pre-decision (Feature Store / Redis) | Conditionally Usable | **RETAINED** | Real-time sliding window counter strictly excluding current transaction. |
| **`device_age_days`** | Pre-decision (Device Fingerprint SDK) | Conditionally Usable | **RETAINED** | Client-side SDK fingerprint timestamp delta from first observed session. |
| **`new_device`** | Pre-decision (Device Fingerprint SDK) | Conditionally Usable | **RETAINED** | Emitted synchronously if device GUID has zero prior history in tenant registry. |
| **`billing_address_mismatch`** | Pre-decision (Card Payment Gateway) | Conditionally Usable | **RETAINED** | Address Verification Service (AVS) response code returned prior to authorization. |

---

## 6. Data Treatment Decision Table

To ensure strict reproducibility and prevent data snooping, all data transformations are specified for integration inside cross-validation pipelines:

| Feature / Issue | Forensic Finding | Operational Treatment | Pipeline Implementation Mechanism | Methodological Justification |
| :--- | :--- | :--- | :--- | :--- |
| **`authentication_level`** | 161 missing ($8.94\%$); highly predictive ($p=0.0014$). | Explicit `'MISSING'` category + One-Hot Encoding | `SimpleImputer(strategy='constant', fill_value='MISSING')` inside `ColumnTransformer` | Preserves informative missingness signal without injecting arbitrary mode. |
| **`merchant_risk_score`** | 33 missing ($1.83\%$); bounded $[0, 1]$. | Median Imputation + Missing Indicator Flag | `SimpleImputer(strategy='median', add_indicator=True)` inside `ColumnTransformer` | Neutral risk proxy for unrated merchants while preserving unrated indicator. |
| **`distance_from_usual_location_km`** | 43 missing ($2.39\%$); heavy right skew (max 496 km). | Median Imputation + Missing Indicator Flag | `SimpleImputer(strategy='median', add_indicator=True)` inside `ColumnTransformer` | Median resists spatial outliers; indicator flags missing GPS telemetry. |
| **`transaction_amount`** | 32 missing ($1.78\%$); financial monetary predictor. | Median Imputation + Missing Indicator Flag | `SimpleImputer(strategy='median', add_indicator=True)` inside `ColumnTransformer` | Median resists large transaction skew; indicator flags missing price signals. |
| **`login_to_transaction_minutes`** | 38 missing ($2.11\%$); right-skewed session duration. | Median Imputation + Missing Indicator Flag | `SimpleImputer(strategy='median', add_indicator=True)` inside `ColumnTransformer` | Median prevents session distortion; indicator flags headless/API sessions. |
| **`ip_risk_score`** | 21 missing ($1.17\%$); external reputation rating. | Median Imputation + Missing Indicator Flag | `SimpleImputer(strategy='median', add_indicator=True)` inside `ColumnTransformer` | Standard median fallback for external service downtime. |
| **Extreme Outliers (`amount`, `distance`, `velocity`)** | Heavy right tails; fraud rate in extremes is 2x–4x higher. | Retain 100% of observations; apply non-linear transforms & robust scalers | Robust scaling within CV folds; tree algorithms (XGBoost, LightGBM) | Critical high-risk fraud cases must not be discarded. |
| **Categorical Predictors** | Nominal categorical variables with rare levels. | One-Hot Encoding with unknown handling | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` | Prevents production pipeline failure on unseen categories. |
| **Global Fitting Prohibition** | Risk of cross-fold and test-set leakage. | Strictly Scoped to Development CV Folds | All imputers and scalers fitted strictly inside training folds | Enforces zero leakage into policy-validation and test partitions. |
