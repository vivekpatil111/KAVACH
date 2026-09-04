# 🛡️ KAVACH: Comprehensive Project Report

**Hackathon Track:** Razorpay AI Buildathon 2026 — AI Risk Manager  
**Project Name:** Kavach (कवच) - Automated Dispute Defense & Profit-Optimized Operations  
**Core Focus:** Chargeback Evidence Responder & Return-Risk Scorer  

---

## 1. Executive Summary

E-commerce merchants face a dual threat: direct financial loss from fraudulent chargebacks and the severe operational overhead of fighting them. Traditional fraud models focus solely on blocking transactions at checkout—often with high false-positive rates that freeze legitimate working capital. 

**Kavach** shifts the paradigm from simple *prevention* to **Automated Defense and Profit-Optimization**. It is an end-to-end Machine Learning and Generative AI system that:
1. **Detects:** Scores transaction risk using a robust XGBoost machine learning model (with real-time <15ms FastAPI inference).
2. **Contextualizes:** Cross-references high-risk claims with actual commercial logistics (shipping SLA, delivery proof, product category) and customer feedback.
3. **Defends:** Automatically generates a card-network-ready dispute defense dossier (PDF) with an LLM-crafted narrative using Gemini, reducing manual analyst time from hours to seconds.
4. **Optimizes:** Identifies the exact mathematical risk threshold that maximizes merchant net savings by balancing fraud capture against API costs and working-capital drag.
5. **Return-Risk:** Evaluates the likelihood of product returns based on logistics performance and historical category data.

---

## 2. Architecture & Technology Stack

The project is built entirely from scratch to be a production-ready prototype.

* **Frontend UI:** Streamlit (Native Custom Deep Charcoal & Teal Theme via `config.toml`)
* **Real-time API:** FastAPI + Uvicorn + Pydantic (Strict schema validation)
* **Machine Learning:** XGBoost, Scikit-learn (Imbalanced data handling, SMOTE)
* **Generative AI:** Google Gemini 1.5 Pro (via `google-genai` SDK)
* **Document Generation:** FPDF2 (Automated Dossier formatting)
* **Data Processing:** Pandas, NumPy, Parquet for high-speed I/O

---

## 3. Data Strategy & Simulated Linkage

The primary challenge in building a comprehensive dispute defense system is that no single open-source dataset contains both payment fraud signals and commercial shipping/fulfillment data. Kavach solves this through a documented, intelligent simulation:

### The Datasets
1. **IEEE-CIS Fraud Detection (Kaggle):** Provides real-world payment transaction features (V-features, C-features, email domains, card networks) and the ground-truth fraud label (`isFraud`).
2. **Olist E-Commerce (Kaggle):** Provides deep commercial fulfillment data (shipping estimates vs. actual delivery, product categories, customer reviews, freight values).

### The Linkage Methodology (Amount Parity)
To simulate a real-world payment gateway's backend, Kavach merges these distinct datasets using **Amount Parity**. 
- IEEE-CIS amounts are denominated in USD.
- Olist amounts are in Brazilian Real (BRL).
- We apply an approximate historical FX rate of 1 USD = 3.50 BRL (2016-2018 average) to the USD transactions.
- We then join high-risk payment claims to Olist orders that match the converted transaction value. 
*Note: This is strictly labeled in the UI as a `SIMULATED DEMO LINKAGE`. In a real production environment, a Payment Aggregator would have native, deterministic joins between their payment and merchant fulfillment databases.*

---

## 4. Core System Modules (In-Depth)

Kavach is composed of highly modular, decoupled components located in the `chargeback_defense/` directory.

### 4.1. `model_trainer.py` (The ML Engine)
* **Process:** Loads raw IEEE-CIS data, engineers 36 targeted features (e.g., `amt_log`, `shipping_billing_dist`, velocity checks), handles class imbalance (SMOTE + scale_pos_weight), and trains an XGBoost classifier.
* **Outputs:** Serialized model (`xgb_model.joblib`), pipeline (`preprocessing_pipeline.joblib`), and a rich Markdown summary of metrics (`data_and_model_summary.md`).
* **Performance:** Achieves an impressive ROC-AUC of **0.875** and a PR-AUC of **0.458** (representing a 13.2x lift over random guessing). At an operational threshold of τ=0.70, it catches 36% of all fraud while ensuring 2 out of every 3 flags are genuinely fraudulent (67% Precision).

### 4.2. `app.py` (Real-Time FastAPI Inference)
* **Process:** Loads the XGBoost model precisely once at server startup (eliminating I/O overhead). It exposes a `/v1/score` POST endpoint.
* **Validation:** Uses Pydantic to enforce that all 36 required features are present and correctly typed, returning clear HTTP 422 errors instead of crashing.
* **Performance:** Average response time is **<15ms**, comfortably exceeding the strict <50ms P99 latency requirement for real-time payment gateway integration.

### 4.3. `evidence_builder.py` (Contextual Aggregation)
* **Process:** This is the heart of the "Evidence Responder." It pulls the simulated linkage described above. It aggregates the payment features, shipping SLA timings (Expected vs. Actual Delivery), and customer review scores into a single `Evidence Packet` dictionary.

### 4.4. `narrative_generator.py` (Generative AI Integration)
* **Process:** Interfaces with the Gemini 1.5 Pro API. It takes the highly structured `Evidence Packet` and uses a sophisticated prompt template to generate a professional, assertive dispute defense narrative formatted to align with standard card-network dispute representment conventions. 
* **Design:** It maps specific evidence items to specific dispute reason codes (e.g., if a delivery was confirmed by a courier, it maps to "Merchandise Delivered").

### 4.5. `batch_runner.py` (Asynchronous Processing)
* **Process:** Simulates a nightly Cron job. It identifies all high-risk claims that lack a defense narrative, queries the Gemini API in bulk (with built-in rate-limit delays), and saves the generated narratives back to the dataset.

### 4.6. `pdf_generator.py` (Automated Dossier Creation)
* **Process:** Converts the JSON evidence packets and LLM narratives into formal, network-ready PDFs. It uses `FPDF2` to structure the document with Kavach branding, clear tabular evidence, and the LLM narrative, ready to be immediately submitted to Visa/Mastercard arbiters.

### 4.7. `return_risk_scorer.py` (Secondary Hackathon Objective)
* **Process:** Fulfills the "Return-Risk Scorer" scope gap using available Olist data. Instead of building a completely separate deep-learning model, it uses a smart, transparent heuristic scoring system. It evaluates logistics failures (late deliveries), poor historical product category ratings, and lack of customer communication to generate a 0-100 "Return Risk Score."

### 4.8. `fp_cost_analysis.py` (The Business Value Engine)
* **Process:** ML models are typically evaluated on F1-scores, but merchants evaluate systems on ROI. This module calculates the *False-Positive Capital Cost*. It simulates a scenario where flagged funds are frozen (e.g., a 3-day hold) subject to a merchant's hurdle rate (e.g., 10%). 
* **Optimization:** It computes the exact "Net Savings" at various risk thresholds by subtracting the opportunity cost of frozen legitimate funds and the Gemini API cost from the total value of fraud successfully prevented.

### 4.9. `dashboard.py` (The Operations UI)
* **Process:** A multi-tab Streamlit dashboard designed as a read-only viewer for fraud analysts.
* **Features:** 
    * **Overview:** High-level metrics, ML performance, and a Live FastAPI Latency Simulator.
    * **Claims Queue:** A filterable table of high-risk transactions.
    * **Evidence Packet Viewer:** A detailed drill-down into a specific claim, showing the LLM narrative and a one-click PDF download.
    * **FP Cost Analysis:** Dynamic sliders for Hurricane/Festive season volume spikes, interactive threshold-optimization graphs, and global INR/USD currency toggles.
    * **Return-Risk:** A dedicated view for the secondary Return-Risk logic.

---

## 5. Key Innovations & Hackathon Differentiation

Why does Kavach stand out?

1. **Shift from Prevention to Defense:** Most fraud systems focus solely on declining transactions. Kavach assumes fraud *will* happen and automates the heavily manual, expensive process of fighting the ensuing chargeback (Dispute Representment).
2. **Financial Reality (The Math of False Positives):** Kavach does not blindly chase 99% Recall. It explicitly calculates the financial damage of False Positives (freezing legitimate merchant capital) and provides a mathematical framework for selecting the most profitable operational threshold.
3. **Production-Ready Coding Standards:** The codebase uses strict typing, OOP design patterns, decoupled microservices (FastAPI vs. Streamlit), Pydantic validation, and clean module separation. There are no Jupyter notebooks passing as production code.
4. **Transparent Generative AI:** Gemini is not used as a black-box decision maker. It is strictly constrained to *drafting the narrative* based on commercial fulfillment evidence fields only — fraud risk scores and model signals are explicitly withheld from the prompt and remain in the internal analyst view only. This grounding in structured evidence fields minimizes hallucination risk and prevents internal risk signals from appearing in external dispute submissions.

---

## 6. Conclusion
Kavach proves that by marrying high-speed traditional Machine Learning (XGBoost) with structured Generative AI (Gemini), Payment Aggregators can offer merchants a shield that not only blocks fraud but autonomously fights back to recover stolen revenue.
