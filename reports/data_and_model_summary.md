# Data & Baseline Model Summary: Kavach

**Date:** September 4, 2026  
**Track:** AI Risk Manager · Razorpay AI Buildathon 2026  
**Target Focus:** Kavach: Data & Model Training Summary (Defense-Only)  
**Status:** Baseline Established · Held-Out Evaluated · Multi-Signal Validated

---

## 1. Dataset Provenance & Integrity

Unlike synthetic simulators, this project is built directly on the **real-world, anonymized IEEE-CIS Fraud Detection dataset** (Vesta Corporation payment transactions):

- **Data Source:** Vesta Corporation / IEEE Computational Intelligence Society.
- **Transaction Table:** Real e-commerce payment transactions across product categories (`W`, `H`, `C`, `S`, `R`), major card networks (Visa, Mastercard, Discover, Amex), and masked entity attributes.
- **Identity Table:** Real browser, device, and network telemetry (`DeviceType`, `DeviceInfo`, operating systems, browser versions, and proxy indicators).
- **Merge Logic:** Left-join on `TransactionID` (`train_transaction` left-joined with `train_identity`). Transactions without identity records (75.58%) are strictly preserved with natural nulls/missing indicators, treating lack of browser telemetry as an explicit defensive risk signal.

### Dataset Dimensions:
| Dataset | Rows | Columns | Memory Usage | Fraud Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Train Merged** | 590,540 | 434 | 1,791.7 MB (1.75 GB) | **3.4990%** (20,663 fraud / 569,877 legit) |
| **Test Merged (Unlabelled)** | 506,691 | 433 | 1,700.3 MB (1.66 GB) | Unlabelled (held for demo inference only) |

---

## 2. Temporal Split Boundaries & Class Balance Stability

To avoid the data leakage inherent in random k-fold cross-validation on time-series payment data, splitting is performed **strictly chronologically** along the `TransactionDT` axis (seconds relative to reference origin):

```
0% ────────────────────────── 70% ────────────────────── 85% ────────────────────── 100%
│        TRAIN SET (70%)       │    VALIDATION (15%)      │     HELD-OUT TEST (15%)   │
│   Day 1.0 ➔ Day 120.8        │  Day 120.8 ➔ Day 152.2   │    Day 152.2 ➔ Day 183.0  │
│  N = 413,378 (3.5169% fraud) │ N = 88,581 (3.4341% fraud│   N = 88,581 (3.4804% fra)│
```

### Exact Split Characteristics:
| Split Name | Share | Row Count | `TransactionDT` Range | Calendar Window | Fraud Count | Fraud Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 70.0% | 413,378 | 86,400 – 10,437,996 | Day 1.0 – Day 120.8 (~4.0 months) | 14,538 | **3.5169%** |
| **Validation** | 15.0% | 88,581 | 10,438,003 – 13,151,840 | Day 120.8 – Day 152.2 (~1.0 month) | 3,042 | **3.4341%** |
| **Held-Out Test** | 15.0% | 88,581 | 13,151,880 – 15,811,131 | Day 152.2 – Day 183.0 (~1.0 month) | 3,083 | **3.4804%** |

### Macroeconomic Stability Check:
- Base rate shift ratio ($\text{Max} / \text{Min}$): $3.5169\% / 3.4341\% = \mathbf{1.024\times}$.
- **Result:** Class balance is remarkably stable across the 6-month timeline. The temporal boundary is realistic and prevents future information from contaminating early decision boundaries.

---

## 3. Data Leakage Precautions

Every feature was engineered under strict causal constraints:

1. **Strictly Causal Trailing Velocity:**
   - Trailing card counts (`card_txn_count_24h`, `card_txn_count_7d`) and trailing dollar volume (`card_txn_amt_sum_24h`) iterate strictly chronologically.
   - For transaction $i$ occurring at timestamp $t_i$, the sliding window contains only prior transactions $j < i$ with $t_j < t_i$. The current transaction is appended to the history queue **only after** its own features are computed.
2. **Amount Anomaly Z-Scores (`card_amt_zscore`):**
   - Running statistics ($\sum x, \sum x^2, N$) are maintained causally. The mean $\mu$ and standard deviation $\sigma$ reflect exclusively the card's prior transaction history.
3. **Target Encoding of Product Category (`ProductCD_train_fraud_rate`):**
   - Empirical Bayes smoothed fraud rates per `ProductCD` are computed **strictly from the 70% Train split** ($N = 413,378$).
   - The smoothed rate lookup table is mapped onto Validation and Held-Out Test sets without updating weights on post-cutoff data.
4. **Validation Separation:**
   - Hyperparameter tuning and early stopping rounds ($40$ rounds) monitored exclusively the Validation split.
   - The Held-Out Test split was evaluated exactly once after training was frozen.

---

## 4. Baseline Model & Test Evaluation

### Model Configuration:
- **Algorithm:** Extreme Gradient Boosting (`xgb.XGBClassifier`)
- **Hyperparameters:** `n_estimators=600`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=5`
- **Class Imbalance Damping:** Negative-to-positive ratio in train is $27.43:1$. Using a damped `scale_pos_weight = 5.24` ($\approx \sqrt{27.43}$) to balance sensitivity while preserving well-calibrated posterior probabilities.
- **Early Stopping:** Best iteration reached at tree **475** based on validation PR-AUC.

### Held-Out Test Performance ($N = 88,581$ transactions, 3,083 fraudulent):

| Metric | Measured Value | Random / Base Rate Benchmark | Relative Lift |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Precision-Recall Area)** | **`0.4588`** | `0.0348` (3.48%) | **`13.2× Lift`** |
| **Random Baseline PR-AUC** | **`0.0348`** | `0.0348` (3.48%) | Equal to fraud class rate in test set |
| **ROC-AUC** | **`0.8752`** | `0.5000` (50.0%) | **`+0.3752`** |

### Candidate Threshold Operating Grid:

| Threshold | Precision | Recall | F1-Score | Total Flagged | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 47.04% | 45.90% | 0.4646 | 3,008 | 1,415 | 1,593 | 1,668 |
| **0.70** (Recommended) | **67.19%** | **36.13%** | **0.4699** | **1,658** | **1,114** | **544** | **1,969** |
| **0.85** | 78.90% | 28.87% | 0.4227 | 1,128 | 890 | 238 | 2,193 |
| **0.90** | 83.30% | 25.24% | 0.3874 | 934 | 778 | 156 | 2,305 |

---

## 5. Operating Threshold Selection & Cost Tradeoff Rationale

### Chosen Operating Threshold: $\tau = 0.70$
- **True Positives (Chargebacks Intercepted):** **`1,114`**
- **False Positives (Legitimate Flagged):** **`544`**
- **False Negatives (Missed Chargebacks):** **`1,969`**
- **True Negatives (Frictionless Approvals):** **`84,954`**
- **Precision:** **`67.19%`** | **Recall:** **`36.13%`** | **F1:** **`0.4699`**

### Business Justification for Chargeback Evidence Responders:
In a payment gateway dispute resolution workflow:
1. **False Positives (FP Cost):** Triggering an automated evidence response or placing a temporary payout reserve on an innocent transaction causes operational overhead and merchant friction (investigating courier delivery slips for a valid order). At $\tau = 0.70$, **precision reaches 67.2%** (2 out of 3 flagged transactions are genuine fraud/chargebacks), capping false alarms at only 544 out of 85,498 legitimate orders (**0.64% merchant friction rate**).
2. **False Negatives (FN Cost):** Missed chargebacks lead to financial loss and card scheme dispute ratio penalties. While $\tau = 0.85$ or $0.90$ achieves higher precision (78%–83%), it drops recall to 25%–28%, missing over 70% of disputes. $\tau = 0.70$ captures 1,114 high-value chargebacks while keeping dispute response preparation tightly focused.

---

## 6. Feature Importance Distribution & Single-Feature Dominance Check

A critical failure pattern in naive fraud models is relying on a single dominant feature (e.g. cluster size or one high-cardinality ID), creating a fragile shortcut that collapses when attackers rotate device IDs or VPAs.

### Full Feature Importance Ranking:

| Rank | Feature Name | Gain Share | Signal Category |
| :---: | :--- | :---: | :--- |
| **1** | `has_billing_addr` | **14.86%** | Address & Mismatch |
| **2** | `ProductCD_train_fraud_rate` | **14.47%** | Product Category Risk |
| **3** | `amt_is_round_dollar` | **9.54%** | Amount Anomaly |
| **4** | `card_address_count_C1` | **6.72%** | Historical Entity Hopping |
| **5** | `has_identity_data` | **4.72%** | Device/Identity Presence |
| **6** | `email_count_C13` | **4.55%** | Account Velocity |
| **7** | `has_recipient_email` | **4.24%** | Dropshipping / Gift Card Risk |
| **8** | `addr_card_country_mismatch` | **3.85%** | Cross-Border Discrepancy |
| **9** | `is_mobile_device` | **3.14%** | Device Fingerprint |
| **10** | `time_delta_prev_txn_D2` | **3.08%** | Transaction Timing Velocity |
| **11** | `amt_cents` | **2.87%** | Amount Granularity |
| **12** | `transaction_count_C2` | **2.62%** | Historical Volume |
| **13** | `browser_is_known` | **1.93%** | Identity Verification |
| **14** | `avs_shipping_match` | **1.65%** | AVS Fulfillment Match |
| **15** | `amt_log` | **1.65%** | Monetary Scale |
| **16** | `is_desktop_device` | **1.63%** | Device Form Factor |
| **17** | `os_is_known` | **1.56%** | Operating System Verification |
| **18** | `time_delta_card_creation_D1` | **1.50%** | Account Age |
| **19** | `email_domain_mismatch` | **1.42%** | P/R Email Discrepancy |
| **20** | `ip_proxy_risk_flag` | **1.38%** | Proxy / Anonymizer Flag |
| **21** | `card_txn_count_7d` | **1.09%** | Weekly Card Velocity |
| **22** | `avs_address_match` | **1.00%** | AVS Address Match |
| **23** | `amt_to_card_mean_ratio` | **0.99%** | Relative Spike Ratio |
| **24** | `email_domain_txn_count_24h` | **0.98%** | Email Burst Velocity |
| **25** | `shipping_billing_dist` | **0.94%** | Physical Distance |
| **26** | `total_verification_matches` | **0.90%** | Multi-Factor Match Depth |
| **27** | `card_txn_count_24h` | **0.85%** | Daily Card Velocity |
| **28** | `has_device_info` | **0.82%** | Hardware Telemetry |
| **29** | `card_txn_amt_sum_24h` | **0.82%** | Trailing Dollar Exposure |
| **30** | `card_time_since_last_txn_hours` | **0.74%** | Inter-Transaction Velocity |
| **31** | `hour_of_day` | **0.72%** | Circadian Context |
| **32** | `card_amt_zscore` | **0.72%** | Relative Card Deviation |
| **33** | `avs_billing_match` | **0.64%** | AVS Billing Match |
| **34** | `day_of_week` | **0.50%** | Day Context |
| **35** | `is_night_transaction` | **0.47%** | Off-Hours Flag |
| **36** | `has_shipping_dist` | **0.45%** | Distance Availability |

### Dominance Check Verdict:
- **Top Feature Share:** `has_billing_addr` = **`14.86%`**
- **Dominance Threshold:** **`50.00%`**
- **Status:** **`PASSED`**  
  No single feature exceeds 15% of total importance. The top 5 features represent diverse dimensions (billing presence, product category risk, amount structure, entity hopping, and device fingerprinting), ensuring the model cannot be bypassed simply by spoofing one parameter.

---

## 7. Evidence-Narrative Generation Layer: Olist Commercial Fulfillment

To build an end-to-end chargeback defense dossier, payment fraud predictions must be corroborated with ground-truth commercial fulfillment, shipping logistics, and customer feedback. We incorporate the real-world **Olist Brazilian E-Commerce Dataset** (99,441 unique commercial orders spanning September 2016 to October 2018).

### Olist Dataset Overview
- **Total Unique Orders:** 99,441
- **Calendar Date Range:** 2016-09-04 to 2018-10-17 (772 days)
- **Total Catalog Items:** 112,650
- **Total Gross Merchandise Value:** R$ 16,009,015.58 BRL
- **Average Order Value:** R$ 160.99 BRL (Median: R$ 105.29 BRL)
- **Carrier Delivery Performance:**
  - `DELIVERED_ON_TIME`: 88,644 (89.14%)
  - `DELIVERED_LATE`: 7,826 (7.87%)
  - `IN_TRANSIT`: 1,722 (1.73%)
  - `CANCELED_OR_UNAVAILABLE`: 1,234 (1.24%)
  - `PENDING`: 15 (0.02%)
- **Customer Review Scores:**
  - 5 Stars: 57,002 (57.77%)
  - 4 Stars: 19,045 (19.30%)
  - 3 Stars: 8,133 (8.24%)
  - 2 Stars: 3,129 (3.17%)
  - 1 Star: 11,364 (11.52%)

---

## 8. Demo Linkage Methodology & Currency Normalization

### Currency Mismatch & Historical FX Justification
- **Source Datasets:** IEEE-CIS fraud transactions are denominated in **USD ($)**, while Olist commercial purchases are denominated in **Brazilian Real (R$ / BRL)**.
- **Problem with Raw Numeric Matching:** Comparing raw amounts directly (e.g., matching $150.00 USD to R$ 131.33 BRL) creates misleading tolerance statistics (e.g., claiming a 12.45% delta) because USD and BRL are distinct currencies with an exchange rate roughly between 3:1 and 4:1 during the active dataset period.
- **Historical FX Benchmark Selection:**
  - During the Olist active operational period (2016–2018), historical exchange rates recorded by the **Banco Central do Brasil** and the **Federal Reserve Economic Data (FRED)** series fluctuated between ~3.10 and ~4.20 BRL per USD:
    - **2016 Annual Average:** ~3.48 BRL / USD
    - **2017 Annual Average:** ~3.19 BRL / USD
    - **2018 Annual Average:** ~3.65 BRL / USD
    - **3-Year Composite Average:** ~3.44–3.50 BRL / USD
  - **Selected Rate:** **`1 USD = 3.50 BRL`** (`HISTORICAL_USD_TO_BRL_FX_RATE = 3.50`).
  - This rate is applied before evaluating tolerance: each IEEE-CIS transaction amount ($USD) is multiplied by 3.50 to yield its BRL target before searching for an equivalent Olist commercial order.

### Linkage Matching Protocol
For each high-risk transaction flagged from the held-out IEEE-CIS test split (fraud probability $\ge 0.70$):
1. **Currency Normalization:** Calculate `amt_brl = round(amt_usd * 3.50, 2)`.
2. **Tolerance Window:** Establish a $\pm 15\%$ parity window: $[0.85 \times \text{amt\_brl},\; 1.15 \times \text{amt\_brl}]$.
3. **Candidate Selection:**
   - If one or more Olist orders fall within the window, select randomly among the candidates to preserve diversity across merchants, product categories, and fulfillment outcomes.
   - If no order falls within the window (e.g., extreme outlier amounts), fall back to the global nearest neighbor and flag `within_target_tolerance: false`.
4. **Parity Delta Calculation:**
   $$\text{delta\_pct} = \frac{|\text{matched\_order\_value\_brl} - \text{amt\_brl}|}{\text{amt\_brl}} \times 100$$

### Linkage Performance Results
- **Total Flagged Claims Scored:** 1,613
- **Matched Within Target Tolerance ($\pm 15\%$):** **1,585 / 1,613 (98.26%)**
- **Average Amount Parity Delta:** **9.17%**
- **Median Amount Parity Delta:** **7.82%**
- **Linkage Artifact:** Saved to [`data/processed/demo_claim_evidence_linkage.json`](file:///d:/Docket-Risk/data/processed/demo_claim_evidence_linkage.json).

---

## 9. Evidence Packet Schema & Representation Transparency

### Dual-Currency Schema Contract
To ensure that no conversion happens silently, every evidence packet explicitly encapsulates both original and converted amounts in its `dispute_summary`:

```json
"dispute_summary": {
  "claim_id": "CLM_3489068",
  "ieee_transaction_id": 3489068,
  "disputed_amount_original": {
    "value": 150.0,
    "currency": "USD"
  },
  "disputed_amount_converted": {
    "value": 525.0,
    "currency": "BRL",
    "fx_rate_used": 3.5,
    "fx_rate_period": "2016-2018 historical average approximation (~3.2-3.8 BRL per USD; Banco Central do Brasil / FRED series)"
  },
  "product_category_code": "H",
  "card_network": "visa",
  "card_type": "debit",
  "ground_truth_label": 1
}
```

In `commercial_fulfillment_evidence`, the parity audit is explicitly recorded:
```json
"commercial_fulfillment_evidence": {
  "source_dataset": "Olist Brazilian E-Commerce (Real commercial logistics)",
  "matched_order_id": "74eba88c228b81196e604cffff7b4fe0",
  "matched_order_value_brl": 455.82,
  "target_converted_amount_brl": 525.0,
  "amount_match_delta_pct": 13.18,
  "within_target_tolerance": true,
  "currency_normalization_note": "Disputed USD 150.00 converted to BRL 525.00 @ 3.5 BRL/USD prior to Olist order matching."
}
```

### Representation Integrity & Simulation Disclaimers
Every evidence packet and linkage record includes the mandatory disclaimer:
> **SIMULATED DEMO LINKAGE** — for pipeline demonstration only, not a real matched transaction. IEEE-CIS amounts are denominated in USD, while Olist commercial order amounts are in Brazilian Real (BRL). An approximate historical FX conversion rate of 1 USD = 3.50 BRL (2016-2018 period average) was applied purely to make demo amount-matching meaningful. This linkage is a SIMULATED_DEMO (no real shared transaction identity between datasets), now currency-normalized for a fair comparison. This is an approximation for demonstration purposes and does not claim currency-exact precision; production systems would require contemporaneous daily FX rates or native same-currency data sources.

### Sample Scenarios Generated
Four diverse representative packets have been compiled at [`reports/sample_evidence_packets.json`](file:///d:/Docket-Risk/reports/sample_evidence_packets.json):
1. **Case 1 (`CLM_3489068`):** Delivered on-time (-10.2 days), 5-star review $\rightarrow$ **`CONTEST_CHARGEBACK_WITH_EVIDENCE`** (`FIRST_PARTY_FRIENDLY_FRAUD`). USD $150.00 $\rightarrow$ BRL R$ 525.00 matched to R$ 455.82 (13.18% delta).
2. **Case 2 (`CLM_3490159`):** Delivered late (+18.6 days), 1-star review $\rightarrow$ **`REVIEW_SHIPPING_SLA_BEFORE_REPRESENTMENT`** (`LATE_DELIVERY_DISPUTE`). USD $77.00 $\rightarrow$ BRL R$ 269.50 matched to R$ 293.42 (8.88% delta).
3. **Case 3 (`CLM_3491784`):** Canceled/unfulfilled order, 1-star review citing wrong product $\rightarrow$ **`ACCEPT_CHARGEBACK_OR_ISSUE_REFUND`** (`UNFULFILLED_MERCHANDISE`). USD $994.00 $\rightarrow$ BRL R$ 3,479.00 matched to R$ 3,184.34 (8.47% delta).
4. **Case 4 (`CLM_3490660`):** High-confidence fraud score (0.947), delivered on-time (-14.2 days), 5-star review $\rightarrow$ **`CONTEST_CHARGEBACK_WITH_EVIDENCE`** (`FIRST_PARTY_FRIENDLY_FRAUD`). USD $100.00 $\rightarrow$ BRL R$ 350.00 matched to R$ 342.07 (2.27% delta).
