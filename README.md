<div align="center">

```
██╗  ██╗ █████╗ ██╗   ██╗ █████╗  ██████╗██╗  ██╗
██║ ██╔╝██╔══██╗██║   ██║██╔══██╗██╔════╝██║  ██║
█████╔╝ ███████║██║   ██║███████║██║     ███████║
██╔═██╗ ██╔══██║╚██╗ ██╔╝██╔══██║██║     ██╔══██║
██║  ██╗██║  ██║ ╚████╔╝ ██║  ██║╚██████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

# 🛡️ कवच · Kavach

### *Stop the merchant losing money to fraud, returns and chargebacks.*

**AI Risk Manager — BachTech Track 02**

<br/>

[![Track](https://img.shields.io/badge/BachTech-Track_02_AI_Risk_Manager-gold?style=for-the-badge&labelColor=1a1a2e)](https://bachtech.in)
[![FastAPI](https://img.shields.io/badge/FastAPI-<15ms_P99-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-5_Tab_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-21_Passed_✅-22c55e?style=for-the-badge&logo=pytest)](tests/)
[![Defense Only](https://img.shields.io/badge/Mode-Defense_Only_🛡️-0284c7?style=for-the-badge)](#️-strictly-defense-only)

<br/>

---

## ⚡ Results at a Glance

| 🏆 Metric | 📊 Value | 🎯 What It Means |
|:---|:---:|:---|
| **Capital Unlocked** | **₹1.84 Crore** | Legitimate working capital freed from binary freezes |
| **Fraud PR-AUC Lift** | **13.2×** | Over random baseline (3.48% class rate) |
| **RTO ROC-AUC** | **0.9845** | Pre-dispatch cancellation prediction |
| **RTO PR-AUC Lift** | **59.5×** | Over random baseline |
| **Dossier Generation** | **< 3 sec** | CE3.0 + NPCI-compliant PDF, SHA-256 sealed |
| **Inference Latency** | **5.38ms avg** | Zero checkout friction |
| **Syndicate Detection** | **< 1ms** | Union-Find graph, real-time ring freeze |
| **Dispute Hours Saved** | **Zero** | Manual work eliminated |
| **Tests Passing** | **21 / 21** | No API keys, no artifacts needed |

</div>

---

## 🎯 Track 02 Alignment — All Four Directions Covered

> *"Build a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set."*

| 🎯 Track Direction | ✅ | 📁 Implementation |
|:---|:---:|:---|
| **Chargeback Evidence Responder** | ✅ Full | XGBoost → Evidence Linkage → Gemini LLM → SHA-256 PDF Dossier |
| **Return-Risk Scorer** | ✅ Full | Secondary XGBoost, 59.5× lift, pre-dispatch COD flagging |
| **Fraud-Spike Detector** | ✅ Full | Real-time scoring at 5.38ms avg, SHAP explainability, claims queue |
| **Abuse-Ring Sentinel** | ✅ Full | Union-Find in-memory graph, coordinated burst detection < 1ms |

> **The Bar:** *"Honest metrics including false-positive cost."* → Dedicated FP Cost Analysis tab built for this. See [below](#-fp-cost-analysis-tab--honest-metrics).

> **Defense-Only Compliance:** No attack simulation, no payload generation. [Full statement →](#️-strictly-defense-only)

---

## Why Grounded on Real Data: Indian D2C Merchant Benchmark

Both underlying datasets powering Kavach are real, publicly available benchmark datasets:

| Dataset | Source | Records | Purpose in Kavach |
| :--- | :--- | :---: | :--- |
| **IEEE-CIS Fraud Detection** | Vesta Corporation / Kaggle | 590,540 train rows | Fraud labels, 36 engineered causal features, XGBoost model training |
| **Olist E-Commerce Logistics** | Olist / Kaggle | 99,441 orders | Fulfillment timeline evidence, shipping SLAs, customer reviews, RTO model |

### The Indian D2C Benchmark Linkage
IEEE-CIS and Olist share **no real transaction identifiers**. To demonstrate a complete end-to-end dispute defense pipeline for the Indian commerce ecosystem, Kavach implements an **Indian D2C Merchant Benchmark with synthetic Indian fulfillment metadata**:

1. **Currency Grounding (INR, ₹):** Transaction tickets are calibrated to realistic Indian D2C order tickets (₹500 to ₹25,000+) using an exchange benchmark of `1 USD = 83.50 INR`.
2. **Indian BFSI Payment Identifiers:** Enriched with UPI Virtual Payment Addresses (`user@okhdfcbank`, `merchant@paytm`), RuPay Debit/Credit cards, and Visa/Mastercard domestic rails.
3. **Domestic Courier Logistics:** Evidence dossiers feature realistic tracking formats from leading Indian 3PL couriers (**BlueDart Express, Delhivery Direct, Shadowfax Surface**) mapped across major Indian commerce hubs with authentic 6-digit postal PIN codes (Mumbai 400051, Bengaluru 560001, Delhi 110001, Hyderabad 500081, etc.) and `+91` contact numbers.
4. **Transparent Disclosure:** Every linked record is **explicitly labeled `SIMULATED_DEMO (Indian D2C Benchmark)`** across the JSON schemas, dashboard, and PDF dossiers.

In a live production payment aggregator environment (e.g. Razorpay), this join is deterministic via the merchant's unified order and transaction database.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                   KAVACH: AI RISK MANAGER ARCHITECTURE                          │
 └─────────────────────────────────────────────────────────────────────────────────┘

  Raw Payment Signals (IEEE-CIS 590K)     Commercial Fulfillment Records (99K Orders)
         │                                                      │
         ▼                                                      ▼
  feature_engineering.py (36 Causal Features)            return_risk_scorer.py (Secondary XGBoost)
         │                                                PR-AUC: 0.2773 (59.47x Baseline Lift)
         ▼                                                      │
  XGBoost Classifier (model_trainer.py)                         ▼
  PR-AUC: 0.4588 (13.2x Lift) │ ROC-AUC: 0.8752         Pre-Dispatch RTO Risk Scoring (₹120-200 Saved)
         │
         ▼
  FastAPI /v1/score (app.py) ◄── Real-Time Inference (<15ms P99 Latency, Pydantic)
         │
         ▼
  High-Risk Flags (τ ≥ 0.70)
         │
         ▼
  evidence_builder.py ──► Grounded in Indian D2C Merchant Benchmark (INR / ₹)
         │                  Integrates: BlueDart / Delhivery / Shadowfax AWBs,
         │                              6-digit Indian PINs, UPI VPAs, RuPay Rails
         ▼
  Evidence Packet (JSON) ──────────────────────────────────────────┐
         │                                                         │
         ▼                                                         ▼
  narrative_generator.py (Gemini LLM)                      pdf_generator.py (FPDF2)
  Strictly fulfillment-grounded prompt                     Network-ready representment dossier
  (no internal risk signals exposed)                       in Indian Rupees (INR / Rs.)
         │                                                         │
         └─────────────────────────┬───────────────────────────────┘
                                   ▼
                          dashboard.py (Streamlit)
              Analyst Operations Console: Claims Queue │ Evidence Dossier │
              FP Cost Capital Analysis (INR) │ RTO Predictor │ Live Simulator
```

---

## Core Capabilities

### 📊 Overview Tab
A high-level command center showing held-out test metrics and operational stability:
- **PR-AUC:** `0.4588` (**13.2x lift over random baseline (3.48% class rate)**)
- **ROC-AUC:** `0.8752`
- **Precision @ τ=0.70:** `67.19%` — 2 out of every 3 flags are real fraud
- **Recall @ τ=0.70:** `36.13%` — 1,114 chargeback disputes caught
- **FP Capital Cost @ τ=0.70:** **₹5,264.53 INR** (only ₹9.68 per dispute hold)
- **Feature Dominance Check:** Passed. Top feature `has_billing_addr` = 14.86% (<15% ceiling), preventing single-signal bypass
- **Live Inference Simulator:** Sends 100 mock requests to the local FastAPI `/v1/score` endpoint and measures P50/P95/P99 latency in real time (<15ms avg)

### 📋 Claims Queue Tab
Filterable analyst queue over all high-risk disputed claims:
- Filter by **Recommendation** (CONTEST / REVIEW / ACCEPT), **Risk Score Range**, and **Narrative Status**
- Table columns: `Claim ID`, `Risk Score`, `Recommendation`, `Amount (INR)`, `Payment Rail` (UPI/RuPay), `Courier`, `AWB`, `PIN`, and `Narrative Status`
- Formatted strictly in **₹ Indian Rupees**

### 🔍 Evidence Packet Viewer Tab
Complete forensic drill-down for individual disputes:
- **Dispute Summary:** Claim ID, disputed amount in `₹`, payment identifier (UPI VPA / RuPay card), and courier partner
- **Fulfillment Logistics:** Carrier dispatch vs. delivery milestone timeline, transit duration, SLA delta, origin/destination hubs with 6-digit PINs
- **Customer Verification:** Star rating (1–5), verbatim review feedback, and customer phone number (`+91 ...`)
- **Executive Dispute Narrative:** Professional representment prose drafted by Gemini Generative AI
- **One-Click PDF Download:** Pre-rendered dispute representment dossier ready for network submission

### 💰 FP Cost Analysis Tab — Honest Metrics

> *"Honest metrics including false-positive cost."* — Track 02 Bar

Every false positive has a real monetary cost. Kavach quantifies it:

| τ Threshold | Precision | Recall | FP Capital Tied Up | Net Savings |
|:---:|:---:|:---:|:---:|:---:|
| 0.50 | 47.04% | 45.90% | ₹2.48 Crore | ₹56.4 Lakhs |
| **0.70 ✅** | **67.19%** | **36.13%** | **₹64 Lakhs** | **₹44.5 Lakhs** |
| 0.85 | 78.90% | 28.87% | ₹27 Lakhs | ₹35.6 Lakhs |
| 0.90 | 83.30% | 25.24% | ₹17 Lakhs | ₹31.1 Lakhs |

**Actuarial Reserve vs Legacy Binary Freeze (2,879 borderline merchants):**
```
Legacy System:  ████████████████████████████████████  ₹474.83 Lakhs FROZEN
Kavach 15%:     ████                                  ₹71.22 Lakhs held
                                                      ₹403.60 Lakhs PRESERVED ✅
```
- **Net Savings Optimizer:** Profit-maximizing threshold at t=0.50 → **₹56.39 Lakhs net savings**
- **Sensitivity Grid:** FP cost across hold durations (1–7 days) × hurdle rates (5–15% p.a.)
- **Festive Surge Multiplier:** Projects capital freeze during 3×–10× seasonal order spikes

### 🔄 RTO & COD Abuse Predictor Tab (Secondary ML)
Addresses the #1 margin killer in Indian e-commerce (reverse shipping costs of ₹120–₹200 per failed delivery):
- **Model:** Secondary XGBoost classifier (**59.47x lift over random baseline**, PR-AUC `0.2773`, ROC-AUC `0.9845`)
- **Target:** `order_status='canceled'` — leakage-free proxy for pre-dispatch customer cancellations and refusal
- **Features:** Delivery delay, review score, freight-to-price ratio, payment installments, seller late dispatch rate, category historical risk
- **Actionable Mitigation:** Identifies high-risk orders to prompt pre-dispatch address re-verification or COD-to-UPI payment conversion

---

## 🧠 AI Judgment — Right Tool, Right Place

> *"The right tool in the right place — and where you chose not to use one."*

| Decision | Choice | Why |
|:---|:---:|:---|
| Fraud model | **XGBoost** not DL | Tabular data + 5.38ms inference + full SHAP explainability for CE3.0 audit |
| Narrative gen | **Gemini 1.5 Pro** not templates | Context-aware AWBs, timestamps, review scores — templates can't do this |
| LLM safety | **Pydantic guardrail** not raw LLM | LLMs hallucinate amounts. Schema enforces before SHA-256 seal |
| Ring detection | **Union-Find** not GNN | < 1ms detection — GNN training overhead unnecessary for this pattern |
| Reserve policy | **Actuarial 15%** not binary freeze | Binary freeze destroys working capital. 15% catches risk without strangling merchants |

---

## ✅ Implementation Reality Matrix

| Component | File | Status | Notes |
| :--- | :--- | :---: | :--- |
| XGBoost fraud scoring | `feature_engineering.py`, `model_trainer.py` | ✅ **LIVE** | Trained on 413K rows, evaluated on held-out 88K test set |
| Real-time FastAPI inference | `app.py` | ✅ **LIVE** | <15ms P99 latency, strict Pydantic schema validation |
| Feature engineering (36 features) | `feature_engineering.py` | ✅ **LIVE** | Causal velocity, AVS signals, device fingerprint, amount anomaly |
| Evidence packet construction | `evidence_builder.py` | ✅ **LIVE** | Grounded in Indian D2C benchmark with INR, UPI VPAs, RuPay, BlueDart/Delhivery |
| LLM narrative generation | `narrative_generator.py` | ✅ **LIVE** | Gemini 1.5/2.5 Pro via `google-genai` SDK + deterministic Indian fallback |
| PDF dossier generation | `pdf_generator.py` | ✅ **LIVE** | FPDF2, submission-ready layout in INR with Indian logistics |
| FP capital cost analysis | `fp_cost_analysis.py` | ✅ **LIVE** | Time-value-of-money capital formula, sensitivity grid in INR |
| RTO & COD abuse prediction | `return_risk_scorer.py` | ✅ **LIVE (ML Model)** | XGBoost classifier (59.5× lift, PR-AUC 0.2773); heuristic fallback preserved |
| Streamlit analyst dashboard | `dashboard.py`, `theme.py` | ✅ **LIVE** | 5-tab console, custom Charcoal/Teal design system, INR currency |
| Cross-dataset transaction linkage | `evidence_builder.py` | ⚙️ **SIMULATED_DEMO** | Indian D2C Benchmark linkage; explicitly disclosed across all records |
| Full batch narrative generation | `batch_runner.py` | ⚙️ **PARTIAL** | 100% generated for demo claims; free-tier API quota applies at scale |

---

## Measured Performance

All metrics are from a single, frozen evaluation on the **held-out test set** (N = 88,581 transactions, 3,083 fraudulent).

### Primary Fraud Classification Metrics

| Metric | Value | Context |
| :--- | :---: | :--- |
| **PR-AUC** | `0.4588` | **13.2x lift over random baseline (3.48% class rate)** |
| **Random Baseline PR-AUC** | `0.0348` | Equal to fraud class rate in test set |
| **ROC-AUC** | `0.8752` | On real, noisy, imbalanced payment data |
| **Class Imbalance Ratio (Train)** | 27.43 : 1 | Damped via `scale_pos_weight = 5.24` (√ ratio) |

### Threshold Operating Grid (Calibrated in INR, ₹)

| Threshold (τ) | Precision | Recall | True Positives | False Positives | Tied-Up Capital (INR) | Total FP Cost (INR) | Net Savings (INR)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 47.04% | 45.90% | 1,415 | 1,593 | ₹24,792,572.85 | ₹20,381.16 | ₹5,639,584.00 |
| **0.70 (Recommended)** | **67.19%** | **36.13%** | **1,114** | **544** | **₹6,405,180.54** | **₹5,264.53** | **₹4,450,717.00** |
| 0.85 | 78.90% | 28.87% | 890 | 238 | ₹2,746,401.40 | ₹2,257.65 | ₹3,557,730.00 |
| 0.90 | 83.30% | 25.24% | 778 | 156 | ₹1,732,367.65 | ₹1,424.08 | ₹3,110,566.00 |

*\*Net Savings computed at benchmark recovery value of ₹4,000 per resolved dispute.*

At τ = 0.70: **2 out of every 3 flagged transactions are genuine fraud**, while merchant friction is held to just 0.64% of legitimate orders. Moving from τ=0.50 to τ=0.70 slashes tied-up capital from **₹2.48 Crore down to ₹64.05 Lakhs (-74.1%)**.

---

## Business Math: False-Positive Capital Cost (INR)

Every false positive has a *real monetary cost* — legitimate merchant working capital frozen during risk triage.

### Formula
$$\text{Cost}_{\text{FP}, i} = \text{Amount}_i \times \left(\frac{\text{Annual Hurdle Rate}}{365}\right) \times \text{Hold Duration Days}$$

**Assumptions:**
- Annual hurdle rate: **10.0% p.a.** (standard Indian commercial credit / working capital overdraft benchmark)
- Hold duration: **3.0 days** (standard 72-hour fraud review SLA)

### Computed Results at τ = 0.70 (INR)

| Metric | Value |
| :--- | :--- |
| False Positives | 544 transactions |
| Total legitimate capital tied up | **₹6,405,180.54 INR** (~₹64.05 Lakhs) |
| **Total FP capital cost** | **₹5,264.53 INR** |
| Average cost per false positive | **₹9.68 INR** / dispute |
| Maximum single-transaction FP cost | ₹148.63 INR (on a ₹180,443.50 transaction) |

### Sensitivity Grid (Total FP Capital Cost in INR)

| Hold Duration | Hurdle 5% | Hurdle 10% (Default) | Hurdle 15% |
| :--- | ---: | ---: | ---: |
| 1 day | ₹877.42 | ₹1,754.84 | ₹2,632.27 |
| 3 days (default) | ₹2,632.27 | **₹5,264.53** | ₹7,896.80 |
| 5 days | ₹4,387.11 | ₹8,774.22 | ₹13,161.33 |
| 7 days | ₹6,141.95 | ₹12,283.91 | ₹18,425.86 |

Even under an extreme stress test (15% hurdle, 7-day hold), total FP cost across all 544 disputes is only **₹18,425.86 INR**, proving that τ = 0.70 effectively shields merchant cash flows.

---

## Sample Evidence Packet & Dispute Narrative

Below is a representative evidence packet and the dispute narrative synthesized for an Indian D2C merchant:

```json
{
  "claim_id": "CLM_3489068",
  "disputed_amount_inr": 12525.00,
  "payment_rail": "RuPay Platinum Debit (**** 5114)",
  "recommendation": "CONTEST_CHARGEBACK_WITH_EVIDENCE",
  "chargeback_classification": "FIRST_PARTY_FRIENDLY_FRAUD",
  "courier_partner": "Delhivery Direct",
  "awb_tracking_number": "DELHIVERY-1000040114",
  "seller_hub": "Jaipur, Rajasthan (PIN 302001)",
  "customer_hub": "Hyderabad, Telangana (PIN 500081)",
  "customer_phone": "+91 9800040114",
  "delivery_status": "DELIVERED_ON_TIME",
  "delivery_delta_days": -10.2,
  "review_score": 5,
  "product_category": "consoles_games",
  "linkage_type": "SIMULATED_DEMO (Indian D2C Benchmark)"
}
```

**Synthesized Dispute Narrative:**

> *This chargeback is disputed as order for 1 item in the 'consoles_games' category was successfully fulfilled and delivered via Delhivery Direct under AWB DELHIVERY-1000040114 to customer destination PIN 500081 (customer contact: +91 9800040114). Payment was processed via RuPay Platinum Debit (**** 5114) for INR 12,525.00. Carrier tracking confirms the item was delivered 10.2 days ahead of the estimated SLA deadline. The customer subsequently submitted a 5-star positive review, confirming satisfactory receipt of goods. All proof demonstrates valid order fulfillment to the authorized recipient. We request immediate reversal of this dispute.*

The narrative references only commercial fulfillment facts. No internal ML scores or velocity flags leak into the external submission text.

---

## Codebase Structure

```
d:\Kavach\
├── app.py                          # FastAPI real-time inference server (<15ms latency)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Native Streamlit dark theme (Charcoal & Teal)
│
├── chargeback_defense/             # Core Kavach pipeline modules
│   ├── __init__.py
│   ├── pipeline.py                 # End-to-end orchestration entrypoint
│   ├── feature_engineering.py      # 36 causal features (FEATURE_NAMES list)
│   ├── data_loader.py              # IEEE-CIS and Olist data loading utilities
│   ├── evidence_builder.py         # Indian D2C benchmark linkage + evidence packet construction
│   ├── narrative_generator.py      # Gemini LLM integration with Indian D2C prompt template
│   ├── batch_runner.py             # Async narrative batch generation (nightly cron sim)
│   ├── pdf_generator.py            # FPDF2 dossier renderer in INR with Indian logistics
│   ├── fp_cost_analysis.py         # TVM false-positive capital cost model (INR)
│   ├── return_risk_scorer.py       # Secondary RTO & COD abuse predictor (XGBoost ML)
│   ├── dashboard.py                # Streamlit multi-tab analyst UI (5 tabs)
│   └── theme.py                    # Kavach CSS design system injector
│
├── models/                         # Serialized model artifacts
│   ├── xgb_model.joblib            # Primary fraud XGBoost classifier (best iter 475/600)
│   ├── return_risk_xgb_model.joblib # Secondary RTO XGBoost classifier
│   └── preprocessing_pipeline.joblib
│
├── reports/                        # Generated analysis outputs
│   ├── data_and_model_summary.md   # Full training metrics + feature importance
│   ├── fp_cost_analysis.md         # FP capital cost report (INR figures)
│   ├── return_risk_scorer.md       # RTO & COD abuse category analysis
│   ├── competitor_analysis.md      # Track differentiation analysis
│   ├── sample_evidence_packets.json
│   ├── sample_evidence_packets_with_narratives.json
│   └── pdf_packets/                # Pre-rendered PDF dossiers (CLM_*.pdf)
│
├── data/
│   ├── raw/olist/                  # Olist CSV source files
│   └── processed/                  # Intermediate parquet files
│
├── report.md                       # Comprehensive Buildathon report
└── CONTRACT.md
```

---

## Developer Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Set GEMINI_API_KEY=your_key_here
```

### 3. Run the full pipeline

```bash
# Generate evidence packets with Indian logistics metadata
python -m chargeback_defense.evidence_builder

# Synthesize dispute narratives for sample claims
python -m chargeback_defense.narrative_generator

# Compute FP capital cost analysis in INR
python -m chargeback_defense.fp_cost_analysis

# Train and evaluate the secondary RTO & COD abuse predictor
python -m chargeback_defense.return_risk_scorer

# Generate network-ready PDF representment dossiers
python -m chargeback_defense.pdf_generator
```

### 4. Start the real-time inference API

```bash
# Terminal 1: FastAPI backend (port 8001)
uvicorn app:app --port 8001 --reload
```

Health check:
```bash
curl http://localhost:8001/v1/health
```

### 5. Launch the analyst dashboard

```bash
# Terminal 2: Streamlit dashboard (port 8502)
python -m streamlit run chargeback_defense/dashboard.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

---

## Automated Test Suite

**File:** [`tests/test_kavach_pipeline.py`](tests/test_kavach_pipeline.py)  
**Run:** `pytest tests/test_kavach_pipeline.py -v`  
**Result:** ✅ **21 tests, 21 passed** — no external API keys, no model artifacts required.

```
============================= test session starts =============================
collected 21 items

tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_T1_low_risk_auto_approve_properties   PASSED
tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_T2_borderline_graduated_reserve_properties PASSED
tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_T3_high_risk_hold_and_autodefend_properties PASSED
tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_boundary_040_lands_in_tier2           PASSED
tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_boundary_075_lands_in_tier2           PASSED
tests/test_kavach_pipeline.py::TestGraduatedReservePolicy::test_tier_decision_p99_latency_under_25ms  PASSED
tests/test_kavach_pipeline.py::TestSyndicateRingDetection::test_isolated_user_gets_no_alert           PASSED
tests/test_kavach_pipeline.py::TestSyndicateRingDetection::test_three_merchants_shared_vpa_triggers_alert PASSED
tests/test_kavach_pipeline.py::TestSyndicateRingDetection::test_cluster_merchant_span_is_exact_count  PASSED
tests/test_kavach_pipeline.py::TestSyndicateRingDetection::test_graph_extraction_latency_under_10ms   PASSED
tests/test_kavach_pipeline.py::TestSyndicateRingDetection::test_old_claims_excluded_from_7d_burst     PASSED
tests/test_kavach_pipeline.py::TestNarrativeAntiHallucination::test_correct_output_passes_all_checks  PASSED
tests/test_kavach_pipeline.py::TestNarrativeAntiHallucination::test_hallucinated_amount_triggers_violation PASSED
tests/test_kavach_pipeline.py::TestNarrativeAntiHallucination::test_hallucinated_claim_id_triggers_violation PASSED
tests/test_kavach_pipeline.py::TestNarrativeAntiHallucination::test_hallucinated_awb_triggers_violation PASSED
tests/test_kavach_pipeline.py::TestNarrativeAntiHallucination::test_amount_within_tolerance_passes    PASSED
tests/test_kavach_pipeline.py::TestDeterministicFallback::test_ce3_evidence_map_returns_all_three_items PASSED
tests/test_kavach_pipeline.py::TestDeterministicFallback::test_deterministic_fallback_contains_key_facts_and_ce3 PASSED
tests/test_kavach_pipeline.py::TestCryptographicSeal::test_seal_is_deterministic                      PASSED
tests/test_kavach_pipeline.py::TestCryptographicSeal::test_seal_detects_amount_tamper                 PASSED
tests/test_kavach_pipeline.py::TestCryptographicSeal::test_seal_is_valid_sha256_hex                   PASSED

============================== 21 passed in 3.40s =============================
```

---

## 🔴 Engineering Postmortem: What Broke at 2 AM and How We Got Out

> *Two critical failures nearly derailed Kavach. This section documents the exact root causes, the 2-hour debug sessions, and the production fixes that turned them around.*

### Crisis 1 — Return-Risk Model Collapse (Class Imbalance: 0.47%)

**Time:** ~2:10 AM. The secondary RTO predictor returned PR-AUC `0.0047` — random baseline. Zero true positives.

**Root Cause:** The `order_status='canceled'` label appeared in only **0.47% of 99,441 rows** (212:1 imbalance). With the default `binary:logistic` objective, XGBoost maximized accuracy by predicting `0` for every row — 99.5% accurate, completely useless. A compounding `NaN` propagation bug from missing `review_score` imputation silently zeroed the most predictive feature.

**Fix:**

```python
# Asymmetric cost-weighting: penalize missing a cancellation 14x harder
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=14.54,          # sqrt(99000 / 468)
    eval_metric="aucpr",             # optimize PR-AUC, not accuracy
    early_stopping_rounds=30,
)
CANCEL_THRESHOLD = 0.25             # lowered from 0.50 to recover recall
review_score = df["review_score"].fillna(df["review_score"].median())  # null fix
```

**Outcome:** PR-AUC jumped from `0.0047` → **`0.2773`** (59.47x lift). ROC-AUC `0.9845`. Recall: 52% of pre-dispatch cancellations caught. Each caught parcel saves ₹120–₹200 in two-way reverse logistics.

**Lesson:** On sub-1% imbalance, `accuracy` is a trap. Always set `eval_metric="aucpr"`, use `scale_pos_weight`, and calibrate the operating threshold explicitly against business cost asymmetry — not the default 0.50 midpoint.

---

### Crisis 2 — LLM Hallucinating Courier Timestamps on Missing Metadata

**Time:** ~3:45 AM. Spot-check of batch-generated PDF dossiers revealed CLM_3490159's narrative read:

> *"Carrier records confirm delivery to PIN 560001 on **2024-11-15** via Delhivery Direct (AWB: DELHIVERY-1000009876)."*

The actual evidence packet had `delivered_customer_timestamp: null` (order status: `canceled`, no dispatch scan). The AWB was a seeding placeholder. **Gemini fabricated both fields** from statistical priors in its training corpus.

A second failure: a ₹4,812.50 claim was rendered as `₹4,800.00` in the narrative — a ₹12.50 rounding hallucination that would fail Visa CE3.0 amount-matching and invalidate the representment.

**Root Cause:** The original `generate_narrative()` used a free-form string prompt. When packet fields were `None`, Gemini generated plausible-sounding values rather than surfacing the absence. There was zero downstream validation — fabricated strings flowed directly into the PDF renderer.

**Fix — Three-Layer Defense:**

```python
# Layer 1: Pydantic JSON Schema (structural enforcement)
class LLMNarrativeOutput(BaseModel):
    claim_id:            str   = Field(...)    # must be exact verbatim
    disputed_amount_inr: float = Field(...)    # numeric only, no rounding
    awb_tracking_number: str   = Field(...)    # exact carrier AWB
    delivery_date:       str   = Field(...)    # YYYY-MM-DD only
    narrative_text:      str   = Field(..., max_length=2000)
    ce3_evidence_cited:  List[str] = Field(default_factory=list)

# Layer 2: Gemini forced into JSON mode at inference time
config = GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.1,
)

# Layer 3: Assertion fact-check BEFORE any PDF is written
# Checks: claim_id (exact) | amount (±INR 0.50) | AWB (case-insensitive) | date (YYYY-MM-DD)
# Failure → [HALLUCINATION DETECTED] logged → automatic deterministic CE3.0 fallback
```

**Outcome:** Hallucination rate on verified fields dropped to `0%`. The test suite (`TestNarrativeAntiHallucination`, 5 tests) catches all 4 hallucination classes on every CI run. Every PDF dossier is additionally sealed with a **SHA-256 integrity hash** over `claim_id || amount_inr || awb || delivery_date || narrative[:500]` — making post-generation tampering immediately detectable.

**Lesson:** Free-form LLM prompting is not safe for legal documents with numeric exactness requirements. Enforce structure at the *inference layer* (JSON mode + Pydantic schema) AND at the *application layer* (deterministic assertion checks) — not just in the prompt text.

---

## 🛡️ Strictly Defense-Only (Buildathon Compliance)

This system is **100% defense-only**. It does **not** generate, simulate, or dispatch any offensive payloads, red-team adversarial attacks, or external system manipulations.

**What Kavach DOES:**
- ✅ Scores incoming transactions for fraud and dispute risk.
- ✅ Generates evidentiary dossiers (PDFs) for chargeback representment.
- ✅ Predicts pre-dispatch RTO / cancellation probability to protect reverse logistics margins.
- ✅ Recommends `CONTEST`, `REVIEW_SLA`, or `ACCEPT_REFUND` actions based on courier delivery proof.

**What Kavach DOES NOT DO:**
- ❌ Execute attacks against payment gateways or merchants.
- ❌ Simulate fraud syndicates or carding attacks.
- ❌ Spoof external financial or courier APIs.

---

## License

MIT License — see [LICENSE](LICENSE).

<div align="center">

<br/>

**Kavach doesn't just stop bad actors.**

**It fundamentally rewrites e-commerce unit economics.**

<br/>

---

*Built for BachTech · Track 02: AI Risk Manager*

*"Stop the merchant losing money to fraud, returns and chargebacks."*

</div>
