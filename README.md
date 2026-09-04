<div align="center">

# 🛡️ Kavach

### Automated Chargeback Evidence Responder & Return-Risk Scorer

*A production-prototype ML + Generative AI system for merchant dispute defense*

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Real--time_Inference-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Analyst_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ROC--AUC-0.8752-10b981?style=for-the-badge" alt="ROC-AUC">
  <img src="https://img.shields.io/badge/PR--AUC-0.4588_(13.2×_lift)-0284c7?style=for-the-badge" alt="PR-AUC">
</p>

</div>

---

## Track Alignment

| Track Objective | Coverage | Implementation |
| :--- | :---: | :--- |
| **Chargeback Evidence Responder** (Primary) | ✅ Full | XGBoost scoring → evidence linkage → Gemini LLM narrative → PDF dossier |
| **Return-Risk Scorer** (Secondary) | ✅ Included | Heuristic scoring on Olist logistics & category data (`return_risk_scorer.py`) |

This project targets the **AI Risk Manager** track of the Razorpay AI Buildathon 2026. The primary focus is on **post-authorisation defense** — automating the dispute representment process after a chargeback is filed — rather than solely blocking transactions at checkout.

---

## Why Real Data

Both datasets powering Kavach are real, anonymized, publicly available datasets — not synthetic simulations.

| Dataset | Source | Records | Purpose in Kavach |
| :--- | :--- | :---: | :--- |
| **IEEE-CIS Fraud Detection** | Vesta Corporation / Kaggle | 590,540 train rows | Fraud labels, 36 engineered features, XGBoost model training |
| **Olist Brazilian E-Commerce** | Olist / Kaggle | 99,441 orders | Commercial fulfillment evidence, shipping SLA, customer reviews |

### The Demo Linkage

IEEE-CIS and Olist share **no real transaction identifiers**. They are independent datasets from different countries and time periods. To build a complete end-to-end dispute defense pipeline — where a fraud flag must be paired with delivery proof — Kavach uses an **amount-parity linkage**:

1. IEEE-CIS amounts (USD) are converted to BRL using a historical FX rate of **1 USD = 3.50 BRL** (Banco Central do Brasil / FRED composite average for 2016–2018, the Olist active period).
2. High-risk payment transactions are joined to Olist orders whose converted BRL amount falls within a tolerance window.
3. Every linked record is **explicitly labeled `SIMULATED_DEMO`** in the JSON, dashboard, and PDF dossier — this disclosure is present everywhere in the system.

In a real production payment gateway, this join would be deterministic: a single transaction database would contain both the payment event and the fulfillment record. The simulation transparently demonstrates what that pipeline would produce.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                            KAVACH PIPELINE                                      │
 └─────────────────────────────────────────────────────────────────────────────────┘

 IEEE-CIS Dataset (590K rows)
        │
        ▼
 feature_engineering.py ──► 36 causal features (velocity, AVS, device, amount, timing)
        │
        ▼
 XGBoost Classifier (model_trainer.py)
   ROC-AUC: 0.8752  │  PR-AUC: 0.4588  │  Operating threshold: τ = 0.70
        │
        ▼
 FastAPI /v1/score (app.py) ◄── Real-time inference endpoint (<15ms P99 latency)
        │
        ▼
 High-Risk Flags (τ ≥ 0.70: 1,658 transactions on held-out test)
        │
        ▼
 evidence_builder.py ──► Amount-parity join to Olist (USD→BRL via 1:3.50 FX)
        │                  Extracts: delivery dates, SLA delta, review score, category
        ▼
 Evidence Packet (JSON) ─────────────────────────────────────────┐
        │                                                         │
        ▼                                                         ▼
 narrative_generator.py                                   pdf_generator.py
 Gemini 2.5 Flash LLM                                    FPDF2 formatted
 (fulfillment-only context,                              card-network-ready
  no risk signals in prompt)                             PDF dossier
        │                                                         │
        └────────────────────────┬────────────────────────────────┘
                                 ▼
                        dashboard.py (Streamlit)
                  Analyst UI: Claims Queue │ Evidence Viewer │
                  FP Cost Analysis │ Return-Risk │ Live Simulator
```

---

## Core Capabilities

### 📊 Overview Tab
A high-level control panel showing real held-out test metrics:
- **PR-AUC:** `0.4588` (13.2× lift over random baseline)
- **ROC-AUC:** `0.8752`
- **Precision @ τ=0.70:** `67.19%` — 2 in 3 flags are real fraud
- **Recall @ τ=0.70:** `36.13%` — 1,114 chargebacks intercepted
- **FP Capital Cost @ τ=0.70:** $63.05 USD (≈ ₹5,953 at illustrative 94.46 INR/USD rate)
- **Feature Dominance Check:** Top feature `has_billing_addr` = 14.86% — no single feature exceeds 15%, confirming the model cannot be bypassed by spoofing one parameter
- **Live Inference Simulator:** Sends 100 mock requests to the local FastAPI `/v1/score` endpoint and measures P50/P95/P99 latency in real time

### 📋 Claims Queue Tab
A filterable analyst queue over all high-risk transactions:
- Filter by **Recommendation** (CONTEST / REVIEW / ACCEPT), **Risk Score Range**, **Narrative Status**
- Inline metrics: total displayed claims, narratives ready, average risk score
- Sortable table with Amount (USD or INR), Card Network, and Narrative Status

### 🔍 Evidence Packet Viewer Tab
Drill-down into any individual claim:
- **Dispute Summary:** Claim ID, card network, transaction amount, fraud risk score, risk band
- **Fulfillment Timeline:** Purchase → carrier dispatch → delivery vs. estimated deadline
- **Delivery Performance:** On-time / late / canceled, delta in days
- **Customer Feedback:** Star rating (1–5), verbatim review comment
- **Merchant & Item Details:** Product category, seller and customer location, item count, payment installments
- **LLM Dispute Narrative:** Full generated narrative formatted for card-network submission
- **One-click PDF Download:** Pre-rendered dispute dossier ready for representment

### 💰 FP Cost Analysis Tab
Business-value analytics using time-value-of-money modeling:
- **Sensitivity Grid:** Total FP capital cost across hold durations (1–7 days) × hurdle rates (5–15%), displayed in USD or INR
- **Threshold Trade-off Table & Charts:** How precision, recall, FP count, tied-up capital, and FP cost shift across τ = 0.50 to 0.90
- **Net Savings Optimizer:** Configurable `avg_chargeback_value` slider (default $50) computes `Net Savings(τ) = TP(τ) × avg_chargeback_value − FP_Cost(τ) − Flagged(τ) × $0.000134/narrative`
- **Seasonal Capital Impact:** Adjustable volume multiplier (1–10×) projecting capital freeze during high-volume periods (e.g., Diwali)
- **Global INR Toggle:** All cost figures convert via fixed illustrative rate (1 USD = 94.46 INR), clearly labeled as non-live

### 🔄 Return-Risk Tab
A secondary, transparent heuristic scoring layer built on Olist logistics:
- **Return-Risk Score (0–100):** Aggregates late delivery penalty, low product category historical rating, and absence of positive customer feedback
- **Return Proxy Label:** Orders classified by `canceled` status or 1-star review (Olist lacks an explicit return flag — this limitation is disclosed in the UI)
- **Category-Level Risk View:** Breakdown of return-risk by product category

---

## Implementation Reality Matrix

| Component | File | Status | Notes |
| :--- | :--- | :---: | :--- |
| XGBoost fraud scoring | `feature_engineering.py`, `pipeline.py` | ✅ **LIVE** | Real model trained on 413K transactions, evaluated on held-out 88K |
| Real-time FastAPI inference | `app.py` | ✅ **LIVE** | <15ms P99, Pydantic validation, model loaded once at startup |
| Feature engineering (36 features) | `feature_engineering.py` | ✅ **LIVE** | Causal velocity, AVS signals, device fingerprint, amount anomaly |
| Evidence packet construction | `evidence_builder.py` | ✅ **LIVE** | Real Olist fulfillment data, currency-normalized |
| LLM narrative generation | `narrative_generator.py` | ✅ **LIVE** | Gemini 2.5 Flash / Claude 3.5 (provider auto-detected from env) |
| PDF dossier generation | `pdf_generator.py` | ✅ **LIVE** | FPDF2, card-network formatted, one file per claim |
| FP capital cost analysis | `fp_cost_analysis.py` | ✅ **LIVE** | TVM formula, sensitivity grid, threshold optimizer |
| Return-Risk scoring | `return_risk_scorer.py` | ✅ **LIVE** | Heuristic, Olist-based, proxy label clearly disclosed |
| Streamlit analyst dashboard | `dashboard.py`, `theme.py` | ✅ **LIVE** | Multi-tab, INR/USD toggle, live API latency simulator |
| Cross-dataset transaction linkage | `evidence_builder.py` | ⚙️ **SIMULATED_DEMO** | Amount-parity join; labeled explicitly on every linked record |
| Live carrier API integration | — | 🔲 **ROADMAP** | Currently sourced from Olist static delivery records |
| Full batch narrative generation | `batch_runner.py` | ⚙️ **PARTIAL** | 100% generated for demo claims; free-tier API quota applies at scale |

---

## Measured Performance

All metrics are from a single, frozen evaluation on the **held-out test set** (N = 88,581 transactions, 3,083 fraudulent). The test split was evaluated exactly once after training was frozen; it was never used for hyperparameter tuning.

### Classification Metrics

| Metric | Value | Context |
| :--- | :---: | :--- |
| **PR-AUC** | `0.4588` | 13.2× lift over the 3.48% random baseline |
| **ROC-AUC** | `0.8752` | On real, noisy, imbalanced payment data |
| **Best Early-Stop Iteration** | Tree 475 / 600 | Monitored on validation PR-AUC |
| **Class Imbalance Ratio (Train)** | 27.43 : 1 | Damped via `scale_pos_weight = 5.24` (√ ratio) |

### Threshold Operating Grid

| Threshold (τ) | Precision | Recall | F1 | True Positives | False Positives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 47.04% | 45.90% | 0.4646 | 1,415 | 1,593 |
| **0.70 ← recommended** | **67.19%** | **36.13%** | **0.4699** | **1,114** | **544** |
| 0.85 | 78.90% | 28.87% | 0.4227 | 890 | 238 |
| 0.90 | 83.30% | 25.24% | 0.3874 | 778 | 156 |

At τ = 0.70: **2 out of every 3 flagged transactions are genuine fraud/chargebacks**, with a merchant friction rate of only 0.64% (544 / 85,498 legitimate orders).

### Feature Importance (Top 10 of 36)

| Rank | Feature | Gain Share | Signal Category |
| :---: | :--- | :---: | :--- |
| 1 | `has_billing_addr` | 14.86% | Address presence / verification |
| 2 | `ProductCD_train_fraud_rate` | 14.47% | Product category historical fraud rate |
| 3 | `amt_is_round_dollar` | 9.54% | Amount anomaly (round-dollar test orders) |
| 4 | `card_address_count_C1` | 6.72% | Card entity hopping velocity |
| 5 | `has_identity_data` | 4.72% | Device / identity telemetry presence |
| 6 | `email_count_C13` | 4.55% | Account velocity (email domain changes) |
| 7 | `has_recipient_email` | 4.24% | Dropshipping / gift card risk proxy |
| 8 | `addr_card_country_mismatch` | 3.85% | Cross-border discrepancy flag |
| 9 | `is_mobile_device` | 3.14% | Device fingerprint |
| 10 | `time_delta_prev_txn_D2` | 3.08% | Transaction timing velocity |

**Dominance Check: PASSED** — No single feature exceeds 15% gain share. The model distributes signal across five distinct risk dimensions (identity, product, amount, velocity, device), preventing single-signal bypass.

---

## Business Math: False-Positive Capital Cost

Traditional fraud evaluation stops at F1-score. Kavach goes further: every false positive has a *real monetary cost* — legitimate merchant funds tied up during a fraud review hold.

### Formula

```
Cost_FP,i = TransactionAmt_i × (annual_hurdle_rate / 365) × hold_duration_days
```

**Assumptions (configurable in dashboard):**
- Annual hurdle rate: **10.0%** p.a. (working capital revolving credit cost proxy)
- Hold duration: **3.0 days** (standard 72-hour fraud triage SLA)

### Computed Results at τ = 0.70

| Metric | Value |
| :--- | :--- |
| False Positives | 544 transactions |
| Total legitimate capital tied up | **$76,708.75 USD** |
| **Total FP capital cost** | **$63.05 USD** |
| Average cost per false positive | $0.1159 (~11.6 cents/dispute) |
| Maximum single-transaction FP cost | $1.78 (on a $2,161.00 transaction) |

### Sensitivity Grid (Total FP Capital Cost, USD)

| Hold Duration | Hurdle 5% | Hurdle 10% | Hurdle 15% |
| :--- | ---: | ---: | ---: |
| 1 day | $10.51 | $21.02 | $31.52 |
| 3 days (default) | $31.52 | **$63.05** | $94.57 |
| 5 days | $52.54 | $105.08 | $157.62 |
| 7 days | $73.56 | $147.11 | $220.67 |

Even at the most extreme scenario (15% hurdle, 7-day hold), total FP cost across all 544 transactions is only **$220.67** — demonstrating that τ = 0.70 keeps financial drag tightly controlled.

### Net Savings Optimizer

```
Net Savings(τ) = TP(τ) × avg_chargeback_value
               − FP_Cost(τ)
               − Flagged(τ) × avg_llm_cost_per_narrative
```

- `avg_chargeback_value` = **$50.00** (configurable assumption; not derived from data)
- `avg_llm_cost_per_narrative` = **$0.000134** (measured from actual Gemini 2.5 Flash batch run)
- At τ = 0.70: `Net Savings = 1,114 × $50 − $63.05 − 1,658 × $0.000134` = **≈ $55,636.50 USD** in fraud prevented, net of all system costs

The dashboard plots this curve across all thresholds with a configurable chargeback value slider.

---

## Sample Evidence Packet & Narrative

Below is a real evidence packet (condensed) and the dispute narrative generated by Gemini 2.5 Flash using the fixed prompt (commercial evidence only — no internal risk signals included in the external text).

```json
{
  "claim_id": "CLM_3489068",
  "card_network": "visa",
  "disputed_amount_usd": 150.00,
  "recommendation": "CONTEST_CHARGEBACK_WITH_EVIDENCE",
  "chargeback_classification": "FIRST_PARTY_FRIENDLY_FRAUD",
  "delivery_status": "DELIVERED_ON_TIME",
  "delivery_delta_days": -10.2,
  "review_score": 5,
  "product_category": "consoles_games",
  "customer_location": "Porto Alegre, RS",
  "payment_method": "credit_card (7 installments)",
  "linkage_type": "SIMULATED_DEMO"
}
```

**Generated Dispute Narrative (Gemini 2.5 Flash, post-prompt-fix):**

> *This chargeback is disputed as order for 1 item in the 'consoles_games' category was successfully fulfilled and delivered. Payment was processed via credit card across 7 installments. Carrier records confirm the item was delivered to the customer on July 17, 2017, which was 10.2 days before the estimated deadline. The customer subsequently provided a positive 5-star review for the transaction, confirming satisfaction with the merchandise received. All evidence demonstrates that the customer received the merchandise as ordered and confirmed receipt with positive feedback. We contend this dispute is without merit and request reversal of the chargeback.*

The narrative references only commercial fulfillment facts. No model risk scores, velocity flags, or internal ML signals appear in the external dispute text.

---

## Codebase Structure

```
d:\Docket-Risk\
├── app.py                          # FastAPI real-time inference server
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Native Streamlit dark theme (Charcoal & Teal)
│
├── chargeback_defense/             # Core Kavach pipeline modules
│   ├── __init__.py
│   ├── pipeline.py                 # End-to-end orchestration entrypoint
│   ├── feature_engineering.py      # 36 causal features (FEATURE_NAMES list)
│   ├── data_loader.py              # IEEE-CIS and Olist data loading utilities
│   ├── evidence_builder.py         # Amount-parity linkage + evidence packet construction
│   ├── narrative_generator.py      # Gemini / Claude LLM integration, SYSTEM_PROMPT
│   ├── batch_runner.py             # Async narrative batch generation (nightly cron sim)
│   ├── pdf_generator.py            # FPDF2 dossier renderer (Kavach branded)
│   ├── fp_cost_analysis.py         # TVM false-positive cost model + threshold optimizer
│   ├── return_risk_scorer.py       # Secondary return-risk heuristic (Olist-based)
│   ├── dashboard.py                # Streamlit multi-tab analyst UI
│   └── theme.py                    # Kavach CSS design system injector
│
├── models/                         # Serialized model artifacts
│   ├── xgb_model.joblib            # Trained XGBoost classifier (best iter 475/600)
│   └── preprocessing_pipeline.joblib
│
├── reports/                        # Generated analysis outputs
│   ├── data_and_model_summary.md   # Full training metrics + feature importance
│   ├── fp_cost_analysis.md         # FP capital cost report
│   ├── return_risk_scorer.md       # Return-risk category analysis
│   ├── competitor_analysis.md      # Track differentiation analysis
│   ├── sample_evidence_packets.json
│   ├── sample_evidence_packets_with_narratives.json
│   └── pdf_packets/                # Pre-rendered PDF dossiers (CLM_*.pdf)
│
├── data/
│   ├── raw/olist/                  # Olist CSV source files
│   └── processed/                  # Intermediate parquet files
│
├── report.md                       # Project overview report
├── CONTRACT.md
└── .env.example                    # API key configuration template
```

---

## Developer Quickstart

### Prerequisites
- Python 3.10+
- A `GEMINI_API_KEY` (Google AI Studio) **or** `ANTHROPIC_API_KEY`
- IEEE-CIS and Olist datasets downloaded from Kaggle (not included in repo due to size)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 3. Run the full pipeline (train → evidence → narratives → PDFs)

```bash
# Train the XGBoost model (requires IEEE-CIS CSVs in data/raw/)
python -m chargeback_defense.pipeline

# Generate evidence packets and LLM narratives for sample claims
python -m chargeback_defense.batch_runner

# Compute FP capital cost analysis
python -m chargeback_defense.fp_cost_analysis

# Compute Return-Risk scores (requires Olist CSVs in data/raw/olist/)
python -m chargeback_defense.return_risk_scorer
```

### 4. Start the real-time inference API

```bash
# Terminal 1: FastAPI backend (port 8001)
uvicorn app:app --port 8001 --reload
```

Test with curl:
```bash
curl -X POST http://localhost:8001/v1/score \
  -H "Content-Type: application/json" \
  -d '{"has_billing_addr": 1, "ProductCD_train_fraud_rate": 0.035, "amt_is_round_dollar": 0, ...}'

# Health check
curl http://localhost:8001/v1/health
```

### 5. Launch the analyst dashboard

```bash
# Terminal 2: Streamlit dashboard (port 8502)
python -m streamlit run chargeback_defense/dashboard.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

---

## Known Limitations & Roadmap

### Current Limitations

| Limitation | Detail |
| :--- | :--- |
| **SIMULATED_DEMO linkage** | IEEE-CIS and Olist share no real transaction IDs. The amount-parity join is a demo approximation, explicitly disclosed on every record. Production systems would use native gateway transaction-fulfillment joins. |
| **Dataset geographic scope** | IEEE-CIS covers US-centric payment patterns; Olist covers Brazilian e-commerce. Neither contains Indian festive-season volume patterns or UPI/RuPay-specific signals. |
| **No live carrier API** | Delivery proof currently sourced from Olist's static historical records. Production would query live logistics APIs (Delhivery, Shiprocket, FedEx) in real time. |
| **Partial narrative coverage** | Free-tier Gemini API quota (20 req/day) limits bulk narrative generation during demo. All pipeline code is functional; `batch_runner.py` handles full-scale generation with a paid key. |
| **Return-risk proxy label** | Olist has no explicit "returned" flag. The proxy (`canceled` status OR 1-star review) is a conservative approximation, disclosed in the dashboard. |
| **FP chargeback value assumption** | The $50 avg chargeback value in the Net Savings optimizer is an illustrative assumption. Neither source dataset contains resolved chargeback settlement amounts. |

### Roadmap

- [ ] **Live logistics API integration** — Replace static Olist delivery records with real-time carrier API queries (Delhivery, Shiprocket) for live delivery proof
- [ ] **Native same-currency data** — Replace the simulated BRL/USD linkage with an Indian payments dataset (UPI, RuPay) where fraud labels and fulfillment records share a native transaction ID
- [ ] **Full-scale narrative generation** — Extend LLM narrative coverage from sample claims to all flagged transactions in each batch cycle
- [ ] **Webhooks & alert integration** — Push high-confidence CONTEST decisions to merchant Slack/email channels automatically
- [ ] **Reason-code-aware prompting** — Map Visa/Mastercard chargeback reason codes (e.g., 4853, 4863) directly to prompt templates for more precise representment language
- [ ] **Merchant-configurable thresholds** — Per-merchant τ configuration with individual hurdle rate and hold-duration parameters stored in a merchant profile database

---

## License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built for the Razorpay AI Buildathon 2026 · AI Risk Manager Track</sub>
</div>
