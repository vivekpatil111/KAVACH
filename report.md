# 🛡️ KAVACH (कवach): Comprehensive Project Report

**Hackathon Track:** Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager (Indian BFSI & E-Commerce Ecosystem)  
**Project Name:** Kavach (कवच) — Automated Dispute Defense & AI Risk Manager  
**Core Deliverables:** Automated Chargeback Evidence Responder (Primary) & RTO / COD Abuse Predictor (Secondary)  
**Target Ecosystem:** Indian D2C Brands, E-Commerce Marketplaces, and Payment Aggregators  

---

## 1. Executive Summary

Indian direct-to-consumer (D2C) merchants and e-commerce enterprises operate on ultra-lean operating margins. In the Indian digital commerce ecosystem, merchants face a severe three-pronged operational drain:
1. **Friendly Fraud & Chargeback Friction:** Cardholders and UPI users filing false "unauthorized transaction" or "merchandise not received" disputes, costing merchants dispute fees and manual representment overhead.
2. **Working-Capital Lockup (False Positives):** Overly aggressive checkout risk filters that flag or hold legitimate customer funds, creating crippling working-capital drag (at typical 10–12% Indian overdraft / commercial credit hurdle rates).
3. **Return-to-Origin (RTO) Logistics Drag:** Failed Cash-on-Delivery (COD) deliveries and frivolous pre-dispatch customer cancellations that incur ₹120–₹200 in non-recoverable two-way reverse logistics costs per failed parcel.

**Kavach** shifts the paradigm from crude checkout blocking to **Automated Defense and Profit-Optimized Risk Operations**. It is an enterprise-grade Machine Learning and Generative AI system tailored for Indian commerce:
* **Real-Time Fraud Scoring:** Sub-15ms FastAPI scoring powered by an XGBoost model achieving a **PR-AUC of 0.4588 (13.2x lift over random baseline)**.
* **Contextual Indian Fulfillment Linkage:** Links disputed claims to domestic logistics evidence across Indian courier networks (**BlueDart Express, Delhivery Direct, Shadowfax Surface**) with 6-digit Indian PIN codes, AWB tracking, and customer verification.
* **Autonomous Dispute Defense Generation:** Synthesizes formal, network-ready representment dossiers (PDF) with Gemini Generative AI narratives, reducing representment turnaround from 48 hours to under 3 seconds.
* **Working Capital Optimization (INR):** Quantifies the exact time-value-of-money cost of false positives in Indian Rupees (₹), demonstrating that moving from τ=0.50 to τ=0.70 cuts tied-up merchant capital by **₹1.84 Crore (-74.1%)** down to **₹64.05 Lakhs**.
* **RTO & COD Abuse Predictor (Secondary ML):** A dedicated XGBoost model (**PR-AUC 0.2773, 59.47x lift**) identifying high-risk cancellation and COD refusal orders prior to warehouse dispatch.

---

## 2. Architecture & Technology Stack

The project is architected as an asynchronous, decoupled, production-grade microservice suite:

```
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                   KAVACH: AI RISK MANAGER ARCHITECTURE                          │
 └─────────────────────────────────────────────────────────────────────────────────┘

  Raw Payment Signals (IEEE-CIS)          Indian D2C Commercial Logistics (Olist Native)
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

* **Backend / API:** FastAPI, Uvicorn, Pydantic v2 (sub-15ms P99 latency, strict schema enforcement).
* **Machine Learning Engines:** XGBoost, Scikit-learn, Joblib (optimized tree-based classification, SMOTE class balancing).
* **Generative AI:** Google Gemini 1.5/2.5 Pro / Flash via `google-genai` SDK with deterministic Indian D2C representment fallback.
* **Document Engine:** FPDF2 with corporate layout, typography sanitation, and submission-ready A4 formatting.
* **Operations UI:** Streamlit with custom dark Charcoal/Teal styling (`theme.py`), interactive charts, and live latency simulation.

---

## 3. Data Grounding: Indian D2C Merchant Benchmark

To address the reality that no single public dataset combines raw credit card/UPI fraud signals with e-commerce shipping tracking, Kavach models an **Indian D2C Merchant Benchmark with synthetic Indian fulfillment metadata**:

### 1. Currency & Ticket Calibration (INR, ₹)
- All transaction values, capital lockups, dispute values, and savings are computed strictly in **Indian Rupees (INR, ₹)**.
- Benchmark parity rate: `1 USD = 83.50 INR`, re-calibrating ticket sizes to realistic Indian D2C order tickets (₹500 to ₹25,000+).
- The working capital hurdle rate is set to **10.0% p.a.**, reflecting standard Indian commercial bank credit / working capital overdraft facilities.

### 2. Indian Payment Rails & BFSI Identifiers
- Replaces generic identifiers with standard Indian BFSI payment rails:
  - **UPI Virtual Payment Addresses (VPAs):** e.g., `user@okhdfcbank`, `merchant@paytm`, `buyer@icici`.
  - **Domestic Card Rails:** RuPay Platinum Debit, RuPay Credit, Visa India, Mastercard.
  - **Customer Telemetry:** E.164 compliant Indian mobile numbers (`+91 98...`) and valid 6-digit postal PIN codes.

### 3. Domestic Courier Logistics & Hubs
- Integrated with realistic Indian 3PL courier tracking standards:
  - **BlueDart Express:** `BD-10000xxxxx` AWB format.
  - **Delhivery Direct:** `DELHIVERY-10000xxxxx` AWB format.
  - **Shadowfax Surface:** `SFX-10000xxxxx` AWB format.
- Major Indian fulfillment origin/destination hubs with authentic 6-digit PIN codes:
  - Mumbai (400051), Bengaluru (560001), New Delhi (110001), Hyderabad (500081), Chennai (600001), Pune (411001), Ahmedabad (380001), Kolkata (700001), Jaipur (302001).

---

## 4. Core System Modules (In-Depth)

### 4.1. `model_trainer.py` — Primary Fraud Classifier
* **Engineering:** Computes 36 causal features (address verification, card entity hopping velocity `card_address_count_C1`, email domain tenure `email_count_C13`, round-rupee transaction anomaly, device telemetry).
* **Performance:** Evaluated on a frozen, held-out test set (N = 88,581 transactions, 3,083 fraudulent):
  - **PR-AUC:** `0.4588` (**13.2x lift over random baseline of 0.0348**).
  - **ROC-AUC:** `0.8752`.
  - **Operational Precision @ τ=0.70:** `67.19%` — 2 out of every 3 flagged transactions are verified fraud.
  - **Recall @ τ=0.70:** `36.13%` (1,114 fraud disputes caught on test set).
  - **Feature Dominance Check:** Passed. Top feature (`has_billing_addr`) has a gain share of 14.86% (<15% ceiling), preventing single-signal bypass.

### 4.2. `app.py` — Real-Time Inference API
* High-speed FastAPI microservice exposing `/v1/score` and `/v1/health`.
* Loads the trained model and preprocessing pipeline once at cold start into memory.
* Validates incoming transaction payloads via strict Pydantic schemas.
* **Latency:** Benchmarked at **<15ms average latency** with P99 <25ms, fully compliant with payment gateway real-time SLA requirements (<50ms).

### 4.3. `evidence_builder.py` — Evidence Aggregation
* Merges high-risk payment disputes with commercial fulfillment records.
* Structures complete evidentiary dossiers: order timeline, carrier SLA delta, courier AWB tracking, customer review rating, and merchant origin/destination hubs.
* Enforces explicit disclosure flags (`SIMULATED_DEMO` / `Indian D2C Benchmark`) on all generated packets.

### 4.4. `narrative_generator.py` — Generative AI Representment
* Interfaces with Gemini 1.5/2.5 Pro via structured prompt engineering.
* **Critical Guardrail:** The prompt strictly contains *fulfillment evidence only* (carrier timestamps, AWB numbers, delivery status, customer star ratings). Internal fraud risk scores, SHAP values, and model features are strictly withheld to eliminate hallucinations and prevent internal risk logic from leaking into external legal disputes.
* Features a robust deterministic fallback (`synthesize_indian_d2c_narrative()`) ensuring 100% test coverage without external API rate-limit bottlenecks.

### 4.5. `pdf_generator.py` — Network-Ready Representment Dossiers
* Produces clean, professional 1-page A4 PDF dispute packets using `FPDF2`.
* Formatted to meet Visa/Mastercard representment guidelines and NPCI dispute verification requirements:
  - Header with Claim ID and recommendation banner (`CONTEST`, `REVIEW_SLA`, `ACCEPT_REFUND`).
  - Disputed amounts formatted in Indian Rupees (`INR ... / Rs. ...`).
  - Courier AWB tracking, transit duration, SLA delta, and customer star rating.
  - Verbatim executive dispute narrative and compelling representment grounds.

### 4.6. `return_risk_scorer.py` — Secondary RTO & COD Abuse Predictor
* Directly addresses the #1 margin drain in Indian e-commerce: Return-to-Origin (RTO) and COD abuse.
* Trained on e-commerce logistics features: courier delay, customer review score, category historical cancellation rate, freight-to-price ratio, and seller dispatch performance.
* **Target Label:** `order_status='canceled'` (strictly leakage-free proxy; review scores are treated as features, not labels).
* **Performance:** Achieves a **PR-AUC of 0.2773**, representing a massive **59.47x lift over random baseline (0.0047)**, with an ROC-AUC of `0.9845`.
* Identifies vulnerable categories (e.g., consumer electronics, fashion accessories) with pre-dispatch risk scores to prompt payment verification (e.g., converting risky COD to UPI payment links).

### 4.7. `fp_cost_analysis.py` — Working Capital ROI Engine
* Replaces academic F1 scores with real-world balance-sheet impact.
* Implements time-value-of-money (TVM) capital lockup modeling:
  $$\text{FP Cost}_i = \text{Amount}_i \times \left(\frac{\text{Hurdle Rate}}{365}\right) \times \text{Hold Days}$$
* **Threshold Trade-Off in Indian Rupees (₹):**
  - At τ = 0.50: Tied-up capital is **₹2.48 Crore** (1,593 false positives). Total FP capital cost = **₹20,381.16**.
  - At τ = 0.70: Tied-up capital is reduced to **₹64.05 Lakhs** (544 false positives). Total FP capital cost = **₹5,264.53** (only **₹9.68 per false positive**).
  - **Capital Drag Reduction:** Moving from τ=0.50 to τ=0.70 releases **₹1.84 Crore (-74.1%)** in legitimate merchant liquidity!

### 4.8. `dashboard.py` — Operations Console
* 5-tab Streamlit dashboard:
  1. **Overview:** High-level metrics, PR/ROC curves, feature importance, and live FastAPI load simulator.
  2. **Claims Queue:** Searchable, filterable queue displaying Indian payment rails, couriers (BlueDart/Delhivery/Shadowfax), AWBs, and PIN codes.
  3. **Evidence Packet Viewer:** Deep-dive dossier viewer with timeline milestones, SHAP risk signals, customer feedback, and one-click PDF download.
  4. **FP Cost Analysis:** Sensitivity matrix across hold durations and hurdle rates, Diwali festive surge multiplier, and profit-maximizing threshold selector.
  5. **RTO & COD Abuse Predictor:** Secondary ML model metrics (PR-AUC 0.2773, 59.47x lift), correlation analysis, and category risk leaderboards.

---

## 5. Threshold Operating Grid (Held-Out Test Set, INR)

| Threshold (τ) | Precision | Recall | True Positives | False Positives | Tied-Up Capital (INR) | Total FP Cost (INR) | Net Savings (INR)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 47.04% | 45.90% | 1,415 | 1,593 | ₹24,792,572.85 | ₹20,381.16 | ₹5,639,584.00 |
| **0.70 (Recommended)** | **67.19%** | **36.13%** | **1,114** | **544** | **₹6,405,180.54** | **₹5,264.53** | **₹4,450,717.00** |
| 0.85 | 78.90% | 28.87% | 890 | 238 | ₹2,746,401.40 | ₹2,257.65 | ₹3,557,730.00 |
| 0.90 | 83.30% | 25.24% | 778 | 156 | ₹1,732,367.65 | ₹1,424.08 | ₹3,110,566.00 |

*\*Net Savings computed at benchmark recovery value of ₹4,000 per resolved dispute.*

---

## 6. Buildathon Compliance: Strictly Defense-Only

Kavach is **100% defense-only**, adhering strictly to the security constraints of the Razorpay AI Buildathon:
* **No Offensive Automation:** No automated attack generators, credential stuffers, bot simulators, or carding scripts exist in the repository.
* **Read-Only Risk Inspection:** The system evaluates incoming telemetry and internal fulfillment data to protect merchants and banks.
* **Deterministic Guardrails:** LLM prompts are restricted to factual fulfillment synthesis, preventing automated execution of unverified financial actions.

---

## 7. Conclusion

**Kavach** proves that by combining high-speed machine learning (XGBoost) with structured Generative AI (Gemini), Indian D2C merchants and payment aggregators can deploy an intelligent risk defense shield. Rather than surrendering to friendly fraud or choking business growth with excessive false-positive holds, Kavach automates dispute representment, protects merchant working capital, and cuts reverse logistics losses before parcels leave the warehouse.

---

## 8. Automated Test Suite

**File:** `tests/test_kavach_pipeline.py`  
**Run:** `pytest tests/test_kavach_pipeline.py -v`  
**Result:** **21 tests, 21 passed (3.40s)** — zero external API calls required.

| Test Class | Count | What is Verified |
| :--- | :---: | :--- |
| `TestGraduatedReservePolicy` | 6 | 3-tier reserve policy thresholds, boundary conditions, and P99 <25ms latency |
| `TestSyndicateRingDetection` | 5 | Union-Find ring detection, merchant span counting, 7-day burst window, <10ms graph latency |
| `TestNarrativeAntiHallucination` | 5 | Pydantic schema pass/fail on all 4 assertion fields (claim_id, amount ±INR 0.50, AWB, delivery_date) |
| `TestDeterministicFallback` | 2 | CE3.0 evidence map structure + deterministic narrative content and latency |
| `TestCryptographicSeal` | 3 | SHA-256 determinism, tamper detection, 64-hex-char format |

---

## 9. Engineering Postmortem: "What Broke at 2 AM and How We Got Out"

> *This section documents two critical production-blocking failures encountered during buildathon development, the root-cause analysis, and the exact fixes applied. These are the inflection points where Kavach either worked or didn't.*

### 🔴 Crisis #1 — Return-Risk Model Collapse (Class Imbalance: 0.47%)

**What happened:**

At approximately 2:10 AM, the secondary RTO/COD abuse predictor's training pipeline completed with a PR-AUC of `0.0047` — effectively random baseline performance (equal to the positive class rate). The model had collapsed to predicting `0` for every input. The confusion matrix showed **zero true positives** across 99,441 test records.

**Root Cause:**

The `order_status='canceled'` target label represented only **0.47% of the dataset** — a 212:1 class imbalance ratio. The default XGBoost objective function (`binary:logistic`) minimized cross-entropy across all samples equally, making it overwhelmingly rational to predict `0` (not canceled) for every input and still achieve >99.5% accuracy. The model had learned to cheat.

The compounding factor: an earlier draft of the preprocessing pipeline applied `StandardScaler` to a boolean review-score feature without null imputation, producing `NaN` propagation that silently turned all `review_score`-derived features to zero, removing the single highest-signal feature from the input matrix.

**Fix Applied:**

```python
# 1. Asymmetric cost-weighted thresholding
# scale_pos_weight = sqrt(negative_count / positive_count)
# = sqrt(99,000 / 468) ≈ 14.54
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=14.54,    # penalizes missing a cancellation 14x harder than a false alarm
    eval_metric="aucpr",       # optimize for PR-AUC, not accuracy
    early_stopping_rounds=30,
)

# 2. Lowered classification threshold from 0.50 to 0.25
# At 0.50 threshold on 0.47% base rate, recall was 0%.
# At 0.25 threshold, recall jumped to 52% while precision held at 19%.
CANCEL_THRESHOLD = 0.25

# 3. Null imputation before scaling
review_score = df["review_score"].fillna(df["review_score"].median())
```

**Outcome:** PR-AUC recovered from `0.0047` to **`0.2773`** — a **59.47x lift** over random baseline. ROC-AUC reached `0.9845`. The model now correctly identifies 52% of high-risk cancellations before warehouse dispatch, saving ₹120–₹200 in reverse logistics per caught parcel.

**Lesson:** On extreme imbalance datasets (`<1%` positive rate), `accuracy` is a completely misleading metric. Always optimize `eval_metric="aucpr"`, use `scale_pos_weight=sqrt(ratio)` as a starting point, and set the operating threshold explicitly based on the business cost asymmetry — not the default 0.50.

---

### 🔴 Crisis #2 — LLM Hallucinating Courier Timestamps on Missing Metadata

**What happened:**

At approximately 3:45 AM, the narrative generation pipeline was running against the full batch of 1,613 high-risk claims. Spot-checking the generated PDF dossiers for CLM_3490159 and CLM_3491784 revealed that the Gemini-generated narrative stated:

> *"Carrier records confirm delivery to PIN 560001 on **2024-11-15** via Delhivery Direct (AWB: DELHIVERY-1000009876)."*

The actual carrier timestamp in the evidence packet was `None` (order was in `canceled` state with no dispatch scan). The AWB was a placeholder. Gemini had **fabricated a delivery date and AWB** from its training prior rather than flagging the absence.

A second class of hallucination involved amount rounding: a claim for `₹4,812.50` was rendered as `₹4,800.00` in the narrative body — a `₹12.50` deviation that would invalidate the representment under Visa CE3.0 amount-matching requirements.

**Root Cause:**

The original `generate_narrative()` function used a free-form string prompt with no structured output constraint. Gemini was given partial evidence packets and asked to "write a dispute narrative." When fields were `None`, the model extrapolated from statistical patterns in its pretraining corpus (courier AWB formats, typical delivery windows) and synthesized plausible-but-false values.

There was no downstream validation — the fabricated strings flowed directly into the PDF renderer.

**Fix Applied (Three-Layer Defense):**

```python
# Layer 1: Pydantic JSON Schema Enforcement
# Force Gemini into structured JSON output mode — no free-text bleed allowed.
class LLMNarrativeOutput(BaseModel):
    claim_id:              str   = Field(..., description="Exact claim ID, verbatim.")
    disputed_amount_inr:   float = Field(..., description="INR amount, numeric only.")
    awb_tracking_number:   str   = Field(..., description="Exact AWB from carrier record.")
    delivery_date:         str   = Field(..., description="Delivery date YYYY-MM-DD.")
    narrative_text:        str   = Field(..., max_length=2000)
    ce3_evidence_cited:    List[str] = Field(default_factory=list)

# Layer 2: Gemini JSON Mode (structural constraint at inference time)
config = genai_types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    response_mime_type="application/json",  # forces valid JSON output only
    temperature=0.1,                         # near-deterministic sampling
)

# Layer 3: Assertion Fact-Check BEFORE any PDF is written
def assert_narrative_facts(llm_output, packet):
    violations = []
    # Check 1: claim_id exact string match
    if llm_output.claim_id != expected_claim_id:
        violations.append(f"CLAIM_ID_MISMATCH: ...")
    # Check 2: amount within ±INR 0.50 tolerance
    if abs(llm_output.disputed_amount_inr - raw_amount) > 0.50:
        violations.append(f"AMOUNT_MISMATCH: ...")
    # Check 3: AWB case-insensitive exact match
    if llm_awb != expected_awb:
        violations.append(f"AWB_MISMATCH: ...")
    # Check 4: delivery date YYYY-MM-DD match
    if llm_date != raw_date:
        violations.append(f"DELIVERY_DATE_MISMATCH: ...")
    return len(violations) == 0, violations

# If ANY violation found → log [HALLUCINATION DETECTED] → fallback to deterministic narrative
```

The deterministic fallback `synthesize_indian_d2c_narrative()` constructs the narrative **programmatically** from raw database fields — zero LLM involvement — and has been verified to produce CE3.0-compliant text across all 4 evidence packet archetypes (CONTEST / REVIEW / ACCEPT / default).

**Outcome:** Hallucination rate in production dropped to `0%` on verified fields. The automated test suite (`TestNarrativeAntiHallucination`) now catches all 4 classes of hallucination on every CI run. PDF dossiers are additionally sealed with a **SHA-256 integrity hash** computed over `claim_id || amount_inr || awb || delivery_date || narrative[:500]`, making any post-generation tampering immediately detectable.

**Lesson:** Free-form LLM prompting is not a safe primitive for legal documents with numeric exactness requirements. Enforce structure at the *model inference layer* (JSON mode + Pydantic schema) AND validate at the *application layer* (deterministic assertion checks) — not just at prompt-engineering time.

---
