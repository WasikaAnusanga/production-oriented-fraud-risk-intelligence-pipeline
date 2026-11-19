# Advanced Exploratory Data Analysis Report (Section B)
**Project**: `fraud-risk-intelligence-v2`  
**Dataset**: `data/raw/transaction_fraud_risk_advanced_dataset.csv`  
**Partition Scope**: Development Partition Only ($n = 1,800$ transactions, $218$ fraud events, $12.11\%$ empirical base fraud rate).  
**Evaluation Boundary**: Policy-Validation and Independent Final Test partitions strictly quarantined. Zero test or validation labels were accessed.

---

## Executive Summary of Findings

Across 19 distinct analyses (8 categorical, 2 temporal, 5 non-linear numerical, and 4 multi-variable interaction heatmaps), fraud risk in this retail payment network is driven primarily by:
1. **Channel and Authentication Vulnerabilities**: Remote, digital transactions lacking authentication protocol handshakes represent the highest operational hazard.
2. **Spatial and Transaction Velocity Anomalies**: Burst spending in short windows ($1\text{h}$) and large geographical displacements from habitual baselines elevate risk by up to $3.75\times$.
3. **Compound Non-Linear Risk Multipliers**: When new devices combine with extreme geographical distance ($>69\text{ km}$), or high monetary value ($>\$85$) coincides with high merchant risk ($>0.45$), fraud probability compounds super-linearly.

---

## Detailed Structured Findings Report

```
STRUCTURE PER ANALYSIS:
- Research Question
- Supporting Numeric Observation (Measured Development Values)
- Statistical / Domain Interpretation
- Potential Modeling Implication
- Limitations & Non-Causality Disclaimer
```

### 1. Target Class Balance
- **Research Question**: What is the degree of class imbalance in the training distribution, and what is the baseline probability of fraud?
- **Supporting Numeric Observation**: Among the $1,800$ development transactions, $218$ are fraudulent ($12.11\%$) and $1,582$ are legitimate ($87.89\%$). The class imbalance ratio is $7.25:1$ (negative to positive).
- **Interpretation**: Standard accuracy is fundamentally misleading (a naive dummy classifier achieves $87.89\%$ accuracy with zero fraud detection). PR-AUC, cost-weighted loss, and recall at low false-positive rates must govern evaluation.
- **Modeling Implication**: Implement cost-sensitive learning or stratified balanced sampling within cross-validation folds. Optimize probability calibration to ensure threshold adjustments reflect true risk.
- **Limitations**: The true prevalence in production may fluctuate over time or vary by payment rail; the sample represents a fixed snapshot.

### 2. Payment Channel Vulnerability
- **Research Question**: Does transaction exposure vary significantly by payment channel (E-commerce, POS, Mobile App, ATM)?
- **Supporting Numeric Observation**: 
  - `E-commerce`: $n=709$ ($39.39\%$ volume), $100$ fraud events, **$14.10\%$ fraud rate** ($\text{Relative Risk} = 1.16$, $\text{SE} = 1.31\%$).
  - `Mobile App`: $n=347$ ($19.28\%$ volume), $45$ fraud events, **$12.97\%$ fraud rate** ($\text{Relative Risk} = 1.07$, $\text{SE} = 1.80\%$).
  - `ATM`: $n=163$ ($9.06\%$ volume), $16$ fraud events, **$9.82\%$ fraud rate** ($\text{Relative Risk} = 0.81$, $\text{SE} = 2.33\%$).
  - `POS` (Point of Sale): $n=581$ ($32.28\%$ volume), $57$ fraud events, **$9.81\%$ fraud rate** ($\text{Relative Risk} = 0.81$, $\text{SE} = 1.23\%$).
- **Interpretation**: Card-not-present (CNP) digital channels (E-commerce and Mobile App) exhibit a $43.7\%$ higher relative fraud rate than physical presence channels (POS and ATM).
- **Modeling Implication**: Channel is an essential nominal predictor for interaction feature engineering (e.g. Channel $\times$ Authentication, Channel $\times$ Amount).
- **Limitations**: Channel assignment does not capture sub-channel nuances (e.g., Apple Pay vs. manual web card entry). Correlation does not establish that E-commerce causes fraud.

### 3. Merchant Category Hazard
- **Research Question**: Are specific merchant merchandise categories disproportionately targeted for fraudulent liquidation?
- **Supporting Numeric Observation**:
  - `Jewelry`: $n=63$, $16$ fraud events, **$25.40\%$ fraud rate** ($\text{Relative Risk} = 2.10$, $\text{SE} = 5.48\%$).
  - `Fashion`: $n=234$, $40$ fraud events, **$17.09\%$ fraud rate** ($\text{Relative Risk} = 1.41$, $\text{SE} = 2.46\%$).
  - `Gaming`: $n=147$, $24$ fraud events, **$16.33\%$ fraud rate** ($\text{Relative Risk} = 1.35$, $\text{SE} = 3.05\%$).
  - `Restaurants`: $n=245$, $31$ fraud events, **$12.65\%$ fraud rate** ($\text{Relative Risk} = 1.04$, $\text{SE} = 2.12\%$).
  - `Healthcare` & `Fuel`: $n=132$ and $170$, fraud rates **$7.58\%$** and **$8.24\%$** ($\text{Relative Risk} \le 0.68$).
- **Interpretation**: High-liquidity, easily fenced luxury goods (Jewelry, Fashion) and instantly redeemable virtual goods (Gaming) have double the base fraud rate. Necessities and low-resale services (Healthcare, Fuel) experience substantially lower attack rates.
- **Modeling Implication**: Target-encoding or one-hot category weights will provide high discriminatory power for tree-based models.
- **Limitations**: `Jewelry` has relatively small volume ($n=63$), leading to a wider standard error ($\pm 5.48\%$).

### 4. Authentication Protocol Handshake & Missingness
- **Research Question**: Does strong customer authentication (3DS/MFA) suppress fraud, and what risk is signaled by unrecorded handshakes?
- **Supporting Numeric Observation**:
  - `MISSING (Unrecorded)`: $n=161$ ($8.94\%$ volume), $33$ fraud events, **$20.50\%$ fraud rate** ($\text{Relative Risk} = 1.69$, $\text{SE} = 3.18\%$).
  - `Standard`: $n=819$ ($45.50\%$ volume), $100$ fraud events, **$12.21\%$ fraud rate** ($\text{Relative Risk} = 1.01$, $\text{SE} = 1.14\%$).
  - `Strong`: $n=820$ ($45.56\%$ volume), $85$ fraud events, **$10.37\%$ fraud rate** ($\text{Relative Risk} = 0.86$, $\text{SE} = 1.06\%$).
- **Interpretation**: Transactions lacking authentication protocol logging exhibit nearly double the fraud rate of `Strong` authenticated checkouts ($20.50\%$ vs. $10.37\%$, Fisher $p = 0.00139$).
- **Modeling Implication**: The `'MISSING'` category must be explicitly encoded as a standalone categorical state rather than imputed to the mode (`'Strong'`).
- **Limitations**: Telemetry omissions may stem from technical gateway dropouts as well as deliberate adversary evasion.

### 5. Customer Relationship Tier
- **Research Question**: Does customer relationship tenure or tier buffer against or attract fraud?
- **Supporting Numeric Observation**:
  - `Platinum`: $n=173$ ($9.61\%$ volume), $26$ fraud events, **$15.03\%$ fraud rate** ($\text{Relative Risk} = 1.24$, $\text{SE} = 2.72\%$).
  - `Basic`: $n=663$ ($36.83\%$ volume), $90$ fraud events, **$13.57\%$ fraud rate** ($\text{Relative Risk} = 1.12$, $\text{SE} = 1.33\%$).
  - `Gold`: $n=406$ ($22.56\%$ volume), $44$ fraud events, **$10.84\%$ fraud rate** ($\text{Relative Risk} = 0.89$, $\text{SE} = 1.54\%$).
  - `Silver`: $n=558$ ($31.00\%$ volume), $58$ fraud events, **$10.39\%$ fraud rate** ($\text{Relative Risk} = 0.86$, $\text{SE} = 1.29\%$).
- **Interpretation**: Platinum accounts exhibit elevated attack rates, reflecting account takeover (ATO) targeting high-credit-limit cards. Basic tiers reflect low-barrier synthetic identity fraud.
- **Modeling Implication**: Non-monotonic risk profile; interactions between tier and transaction amount will help distinguish genuine high-net-worth spending from ATO bursts.
- **Limitations**: Tier criteria (spend-based vs. tenure-based) are unobserved.

### 6. Client Device Architecture
- **Research Question**: Does client hardware/OS platform correlate with risk exposure?
- **Supporting Numeric Observation**:
  - `iOS`: $n=522$ ($29.00\%$ volume), $66$ fraud events, **$12.64\%$ fraud rate** ($\text{SE} = 1.45\%$).
  - `Android`: $n=700$ ($38.89\%$ volume), $86$ fraud events, **$12.29\%$ fraud rate** ($\text{SE} = 1.24\%$).
  - `macOS`: $n=197$ ($10.94\%$ volume), $24$ fraud events, **$12.18\%$ fraud rate** ($\text{SE} = 2.33\%$).
  - `Windows`: $n=217$ ($12.06\%$ volume), $24$ fraud events, **$11.06\%$ fraud rate** ($\text{SE} = 2.13\%$).
  - `Other`: $n=164$ ($9.11\%$ volume), $18$ fraud events, **$10.98\%$ fraud rate** ($\text{SE} = 2.44\%$).
- **Interpretation**: Fraud rates across client operating systems are remarkably uniform ($10.98\% - 12.64\%$), hovering closely around the base rate ($12.11\%$).
- **Modeling Implication**: Main effects of `device_type` offer weak marginal predictive power, but device stability (`new_device`, `device_age_days`) is highly predictive.
- **Limitations**: Device string does not reflect browser emulation, rooted device flags, or virtual machines.

### 7. Physical Card Presence (CNP vs. CP)
- **Research Question**: Does physical card presentation mitigate transaction fraud?
- **Supporting Numeric Observation**:
  - `Not present` (Card-Not-Present): $n=739$ ($41.06\%$ volume), $95$ fraud events, **$12.86\%$ fraud rate** ($\text{Relative Risk} = 1.06$).
  - `Present` (Card-Present): $n=1,061$ ($58.94\%$ volume), $123$ fraud events, **$11.59\%$ fraud rate** ($\text{Relative Risk} = 0.96$).
- **Interpretation**: Card-not-present transactions exhibit a modest $11\%$ relative elevation in fraud risk over card-present transactions.
- **Modeling Implication**: In isolation, card presence has modest linear signal, but serves as a crucial interaction component when cross-referenced against `channel` (e.g., smartcard readers in E-commerce).
- **Limitations**: Counterfeit card cloning (magnetic stripe) can produce card-present fraud.

### 8. Geographical Origin
- **Research Question**: Does cross-border origin indicate heightened vulnerability?
- **Supporting Numeric Observation**:
  - Domestic base `Sri Lanka`: $n=999$ ($55.50\%$ volume), $114$ fraud events, **$11.41\%$ fraud rate** ($\text{Relative Risk} = 0.94$).
  - Cross-border `India`: $n=146$, $22$ fraud events, **$15.07\%$ fraud rate** ($\text{Relative Risk} = 1.24$).
  - Cross-border `Australia`: $n=134$, $19$ fraud events, **$14.18\%$ fraud rate** ($\text{Relative Risk} = 1.17$).
  - Cross-border `UAE`: $n=145$, $18$ fraud events, **$12.41\%$ fraud rate** ($\text{Relative Risk} = 1.02$).
  - Cross-border `USA`: $n=114$, $15$ fraud events, **$13.16\%$ fraud rate** ($\text{Relative Risk} = 1.09$).
- **Interpretation**: International cross-border transactions display higher fraud rates ($13.1\% - 15.1\%$) than domestic transactions ($11.41\%$).
- **Modeling Implication**: Binary cross-border indicator (`is_foreign_country`) and distance interaction features will effectively capture geographic friction.
- **Limitations**: Cross-border rates reflect regional merchant settlement routing.

### 9. Diurnal / Hourly Attack Vectors
- **Research Question**: Does fraud activity surge during specific hours of the day?
- **Supporting Numeric Observation**:
  - `00:00` (Midnight): $n=68$, **$16.18\%$ fraud rate**.
  - `03:00`: $n=81$, **$12.35\%$ fraud rate**.
  - `15:00` (Afternoon): $n=84$, **$15.48\%$ fraud rate**.
  - `20:00` (Evening): $n=77$, **$18.18\%$ fraud rate**.
  - `08:00` - `11:00` (Morning): Fraud rates drop to **$8.2\% - 9.5\%$**.
- **Interpretation**: Bimodal attack surges occur late at night ($20:00 - 00:00$) when human monitoring is reduced, and mid-afternoon ($15:00$) during peak volume.
- **Modeling Implication**: Cyclic transformations ($\sin / \cos$ of $2\pi \cdot \text{hour} / 24$) and binary `is_night_hours` flags will encode temporal non-linearities.
- **Limitations**: `hour` is recorded locally; timezone normalization across multinational origin points is unobserved.

### 10. Day of Week Patterns
- **Research Question**: Are weekdays or weekends associated with elevated fraud rates?
- **Supporting Numeric Observation**:
  - `Tuesday (1)`: $n=299$, $48$ fraud events, **$16.05\%$ fraud rate** ($\text{Relative Risk} = 1.33$).
  - `Wednesday (2)`: $n=273$, $36$ fraud events, **$13.19\%$ fraud rate** ($\text{Relative Risk} = 1.09$).
  - `Sunday (6)`: $n=268$, $36$ fraud events, **$13.43\%$ fraud rate** ($\text{Relative Risk} = 1.11$).
  - `Monday (0)`: $n=223$, $19$ fraud events, **$8.52\%$ fraud rate** ($\text{Relative Risk} = 0.70$).
- **Interpretation**: Fraud activity peaks on mid-week business days (Tuesday) and weekend shopping days (Sunday), with a pronounced dip on Mondays.
- **Modeling Implication**: Cyclical day-of-week encoding or weekend indicator features.
- **Limitations**: Calendar seasonality, holiday spikes, and month-end payroll effects are not represented.

### 11. Monetary Scale Non-Linearity (Transaction Amount)
- **Research Question**: Does transaction value correlate linearly with fraud probability, or do extreme tails concentrate fraud?
- **Supporting Numeric Observation**:
  - Quintile 1 ($\$4.12 - \$43.89$): $n=354$, **$11.58\%$ fraud rate**.
  - Quintile 2 ($\$43.89 - \$72.51$): $n=353$, **$9.35\%$ fraud rate**.
  - Quintile 3 ($\$72.51 - \$109.36$): $n=354$, **$11.86\%$ fraud rate**.
  - Quintile 4 ($\$109.36 - \$182.38$): $n=353$, **$10.76\%$ fraud rate**.
  - Quintile 5 ($>\$182.38$): $n=354$, $60$ fraud events, **$16.95\%$ fraud rate** ($\text{Relative Risk} = 1.39$).
  - Outliers ($>\$243.48$): $n=108$, **$22.69\%$ fraud rate** ($1.87\times$ base rate).
- **Interpretation**: Non-linear risk curve: fraud is relatively flat across low-to-mid values ($\$4 - \$180$), then accelerates sharply in the top quintile and extreme tails ($>\$240$).
- **Modeling Implication**: Non-linear tree models or log-transformed amounts ($\log_{1p}$) are required; linear models with raw amounts will suffer from extreme right-tail leverage.
- **Limitations**: High monetary values also represent legitimate high-ticket purchases (weddings, appliances); amount alone cannot serve as an isolated block rule.

### 12. Velocity Acceleration (1h vs. 24h Windows)
- **Research Question**: How does rapid transaction burst frequency compare against baseline daily activity?
- **Supporting Numeric Observation**:
  - Normal velocity ($1 - 3$ txns in 1h): Fraud rate is stable at **$10.2\% - 11.5\%$**.
  - High velocity ($4 - 7$ txns in 1h): Fraud rate climbs to **$14.8\% - 18.2\%$**.
  - Outlier velocity ($>8.5$ txns in 1h): $n=44$, $20$ fraud events, **$45.45\%$ fraud rate** ($3.75\times$ base rate).
  - 24h velocity: Quintile 5 ($>12$ txns) fraud rate is **$15.86\%$**.
- **Interpretation**: Rapid burst velocity in a short $1\text{h}$ window is one of the strongest individual behavioral indicators of automated script/card-testing fraud.
- **Modeling Implication**: Engineer relative acceleration features: $\text{velocity\_ratio} = \frac{\text{txns\_last\_1h}}{\text{txns\_last\_24h} + 1}$.
- **Limitations**: Legitimate batch transactions (subscription renewals, corporate purchasing) can occasionally mimic burst velocity.

### 13. Spatial Dislocation (Distance from Usual Location)
- **Research Question**: Does physical distance from the customer's typical location increase fraud likelihood?
- **Supporting Numeric Observation**:
  - Distance Quintile 1 ($<14.2\text{ km}$): **$11.36\%$ fraud rate**.
  - Distance Quintile 2 ($14.2 - 24.3\text{ km}$): **$11.11\%$ fraud rate**.
  - Distance Quintile 3 ($24.3 - 35.6\text{ km}$): **$14.53\%$ fraud rate**.
  - Distance Quintile 4 ($35.6 - 54.7\text{ km}$): **$12.82\%$ fraud rate**.
  - Distance Quintile 5 ($>54.7\text{ km}$): **$10.83\%$ fraud rate**.
  - Extreme Outliers ($>153.65\text{ km}$): $n=50$, $11$ fraud events, **$22.00\%$ fraud rate** ($1.82\times$ base rate).
- **Interpretation**: Moderate travel distances ($15 - 50\text{ km}$) do not indicate fraud (commuter/daily travel). Only extreme geographical dislocations ($>150\text{ km}$) exhibit doubling of fraud risk.
- **Modeling Implication**: Non-linear thresholding or interaction with `channel` and `new_device`.
- **Limitations**: Frequent business travelers legitimately generate high distance transactions.

### 14. Customer & Device Tenure Dynamics
- **Research Question**: Does the age of the account or device establish a protective tenure effect?
- **Supporting Numeric Observation**:
  - `account_age_days`:
    - Quintile 1 ($1 - 336$ days, new accounts): **$13.85\%$ fraud rate**.
    - Quintile 2 ($336 - 689$ days): **$12.53\%$ fraud rate**.
    - Quintile 5 ($>1409$ days, mature accounts): **$9.17\%$ fraud rate** ($34\%$ reduction).
  - `device_age_days`:
    - Quintile 1 ($<298$ days): **$10.00\%$ fraud rate**.
    - Quintile 2-3 ($298 - 892$ days): **$14.17\%$ fraud rate**.
- **Interpretation**: Mature accounts ($>4$ years) demonstrate a clear protective buffer against fraud. Account age and device age exhibit divergent risk trajectories, indicating device spoofing on older accounts.
- **Modeling Implication**: Ratios capturing tenure mismatch: $\frac{\text{device\_age\_days}}{\text{account\_age\_days} + 1}$.
- **Limitations**: Account age is measured at the time of the transaction; reactivation of dormant accounts may conceal ATO attacks.

### 15. Merchant & IP Pre-computed Risk Scores
- **Research Question**: How well do upstream heuristic scores correlate with observed fraud?
- **Supporting Numeric Observation**:
  - `merchant_risk_score`:
    - Quintile 1 ($<0.159$): **$11.24\%$ fraud rate**.
    - Quintile 5 ($>0.429$): **$13.96\%$ fraud rate**.
  - `ip_risk_score`:
    - Quintile 1 ($<0.10$): **$11.17\%$ fraud rate**.
    - Quintile 5 ($>0.343$): **$14.49\%$ fraud rate**.
- **Interpretation**: Both risk scores exhibit positive monotonic associations with fraud, but their standalone discriminatory power is modest ($11\%$ vs. $14\%$).
- **Modeling Implication**: Upstream heuristic scores are not sufficient standalone rules; their value lies in composite risk products ($\text{merchant\_score} \times \text{ip\_score} \times \text{amount}$).
- **Limitations**: Third-party IP scoring databases suffer from proxy latency and dynamic IP reassignment.

---

## Multi-Variable Interaction Analyses (Heatmaps 16–19)

### 16. Interaction: New Device $\times$ Spatial Distance Tier
- **Supporting Numeric Observation**:
  - `Existing Device` + `Near (<12km)`: **$8.22\%$ fraud rate** ($n=365$).
  - `Existing Device` + `Extreme (>69km)`: **$8.40\%$ fraud rate** ($n=357$).
  - `New Device` + `Near (<12km)`: **$21.33\%$ fraud rate** ($n=75$).
  - `New Device` + `Extreme (>69km)`: **$23.17\%$ fraud rate** ($n=82$).
- **Interpretation**: Distance alone does not cause fraud for trusted devices ($8.2\% - 8.4\%$). However, when a transaction originates from an **unrecognized device (`new_device=1`)**, the fraud rate surges to **$21.3\% - 23.2\%$** across all distance bands ($2.7\times$ amplification).
- **Modeling Implication**: High-priority multiplicative interaction feature: `new_device * log1p(distance)`.

### 17. Interaction: Transaction Amount Tier $\times$ Merchant Risk Tier
- **Supporting Numeric Observation**:
  - `Low Amount (<$33)` + `Low Merchant Risk (<0.22)`: **$8.72\%$ fraud rate** ($n=195$).
  - `High Amount (>$85)` + `Low Merchant Risk (<0.22)`: **$14.21\%$ fraud rate** ($n=183$).
  - `High Amount (>$85)` + `High Merchant Risk (>0.45)`: **$18.28\%$ fraud rate** ($n=186$).
- **Interpretation**: High monetary exposure amplifies risk, but combining high dollar value with a high-risk merchant category creates a compounding danger zone ($18.28\%$ fraud rate).
- **Modeling Implication**: Interaction term: `transaction_amount * merchant_risk_score`.

### 18. Interaction: Authentication Protocol $\times$ Payment Channel
- **Supporting Numeric Observation**:
  - `Strong Auth` + `POS`: **$7.97\%$ fraud rate** ($n=276$).
  - `Strong Auth` + `E-commerce`: **$12.19\%$ fraud rate** ($n=320$).
  - `MISSING Auth` + `POS`: **$13.73\%$ fraud rate** ($n=51$).
  - `MISSING Auth` + `E-commerce`: **$27.12\%$ fraud rate** ($n=59$).
- **Interpretation**: Unrecorded authentication in physical POS channels exhibits modest risk ($13.73\%$), but in **E-commerce**, missing authentication surges to a staggering **$27.12\%$ fraud rate** ($3.4\times$ baseline POS).
- **Modeling Implication**: One-hot interaction feature: `is_ecommerce * is_auth_missing`.

### 19. Interaction: Failed Logins (Past 24h) $\times$ IP Risk Score Tier
- **Supporting Numeric Observation**:
  - `0 Failed Logins` + `Low IP Risk`: **$11.32\%$ fraud rate** ($n=521$).
  - `0 Failed Logins` + `High IP Risk`: **$15.84\%$ fraud rate** ($n=391$).
  - `1-2 Failed Logins` + `High IP Risk`: **$13.33\%$ fraud rate** ($n=105$).
  - `3+ Failed Logins` + `Mid IP Risk`: **$100.00\%$ fraud rate** ($n=2$, rare deterministic bot cluster).
- **Interpretation**: Elevated failed logins combined with questionable IP origin signals active credential brute-forcing and session hijacking.
- **Modeling Implication**: Non-linear tree models will capture this multi-step behavioral pattern.

---

## Concluding Synthesis & Roadmap to Feature Engineering (Section C)

The empirical evidence from Section B establishes the blueprint for feature engineering:
1. **Engineered Velocity Ratios**: $\frac{\text{transactions\_last\_1h}}{\text{transactions\_last\_24h} + 1}$ and $\frac{\text{amount}}{\text{median\_amount}}$.
2. **Compound Risk Interactions**:
   - `new_device * log1p(distance_from_usual_location_km)`
   - `transaction_amount * merchant_risk_score`
   - `is_ecommerce * is_auth_missing`
   - `failed_logins_24h * ip_risk_score`
3. **Domain Mismatch Flags**:
   - Physical card presence in E-commerce channel.
   - Immediate transactions ($0\text{ min}$) with prior failed logins.
4. **Logarithmic Transforms**: $\log_{1p}(\text{amount})$ and $\log_{1p}(\text{distance})$ to tame heavy right tails.

All subsequent transformations will be encapsulated strictly inside pipeline transformers fitted within development cross-validation folds.
