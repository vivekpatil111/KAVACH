# 🔄 RTO (Return-to-Origin) & COD Abuse Predictor: Indian D2C Risk Analysis

**Hackathon Track:** Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager  
**Module:** `return_risk_scorer.py` & `train_return_risk_model.py`  
**Model Engine:** XGBoost  
**Held-out Performance:** PR-AUC 0.2773 (59.48x lift over 0.004662004662004662 random baseline) · ROC-AUC 0.9845  

---

## 1. The Indian D2C Problem: Return-to-Origin (RTO) & COD Abuse

In the Indian e-commerce ecosystem, Cash-on-Delivery (COD) remains a dominant payment method (often exceeding 60% of volume in Tier-2 and Tier-3 markets). However, COD introduces severe merchant friction:
- **RTO Failure Rates:** 20% to 35% of COD orders end in Return-to-Origin due to customer refusal, fictitious addresses, or impulse cancellations.
- **Reverse Logistics Drag:** Each RTO event incurs both forward and reverse shipping costs (averaging **₹120 – ₹200 per failed shipment** through logistics partners like Delhivery, BlueDart, and Shadowfax), completely wiping out gross margins.
- **COD-to-UPI Arbitrage / Abuse:** Fraudulent buyers exploit COD ordering for speculative purchases or abuse refund loops.

**Kavach's Solution:** Pre-dispatch prediction of RTO and cancellation risk using logistics telemetry, freight-to-price ratios, seller fulfillment lag, and historical category risk. High-risk orders trigger automated merchant playbooks:
1. **Incentivize Prepayment:** Offer an immediate 5% discount to convert high-risk COD orders to instant UPI payments via Razorpay.
2. **Automated Verification:** Trigger an automated WhatsApp/SMS interactive address verification before booking courier dispatch.
3. **Tier-1 Logistics Routing:** Route borderline orders exclusively through premium couriers (BlueDart Express / Delhivery Direct) with OTP-verified delivery.

---

## 2. Model Performance Summary

| Metric | Value | Benchmark Context |
| :--- | :---: | :--- |
| **PR-AUC** | `0.2773` | **59.48x lift** over random baseline |
| **Random Baseline PR-AUC** | `0.004662004662004662` | Cancellation rate in held-out test split |
| **ROC-AUC** | `0.9845` | Strong discriminatory ability across logistics signals |
| **5-Fold CV PR-AUC** | `0.2869 ± 0.0293` | Stable cross-validated training performance |
| **Recall @ τ=0.50** | `90.22%` | Intercepts 90%+ of high-risk cancellations |
| **Delivery Delay Correlation (r)** | `0.3521` | Positive correlation confirms shipping delay drives return spikes |

---

## 3. High-RTO Risk Categories (Indian D2C Benchmark)

Categories with highest average ML cancellation/RTO probability scores:

| Category | Avg ML Risk Score | Proxy Return Rate | Avg Review Score | Avg Delay (Days) | Orders |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `furniture_bedroom` | 0.0499 | 11.70% | 4.16 | -10.2 | 94 |
| `construction_tools_safety` | 0.0459 | 17.28% | 3.88 | -10.1 | 162 |
| `books_imported` | 0.0434 | 9.62% | 4.38 | -8.4 | 52 |
| `dvds_blu_ray` | 0.0401 | 15.25% | 4.07 | -10.3 | 59 |
| `fashion_male_clothing` | 0.0396 | 23.21% | 3.70 | -10.4 | 112 |
| `unknown` | 0.0351 | 16.21% | 3.91 | -9.0 | 1,437 |
| `art` | 0.0330 | 11.62% | 4.04 | -10.5 | 198 |
| `home_appliances_2` | 0.0324 | 9.87% | 4.13 | -10.4 | 233 |
| `consoles_games` | 0.0320 | 11.25% | 4.06 | -9.1 | 1,058 |
| `small_appliances` | 0.0320 | 11.00% | 4.17 | -11.7 | 627 |

---

## 4. Safest Low-RTO Categories

| Category | Avg ML Risk Score | Proxy Return Rate | Avg Review Score | Avg Delay (Days) | Orders |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `costruction_tools_tools` | 0.0001 | 8.25% | 4.38 | -11.5 | 97 |
| `tablets_printing_image` | 0.0004 | 5.19% | 4.14 | -12.4 | 77 |
| `audio` | 0.0050 | 16.47% | 3.83 | -9.1 | 346 |
| `office_furniture` | 0.0054 | 17.87% | 3.62 | -10.4 | 1,265 |
| `construction_tools_lights` | 0.0060 | 6.44% | 4.17 | -10.2 | 233 |
| `industry_commerce_and_business` | 0.0079 | 8.62% | 4.19 | -10.8 | 232 |
| `books_technical` | 0.0079 | 7.34% | 4.38 | -10.0 | 259 |
| `electronics` | 0.0081 | 10.79% | 4.09 | -9.7 | 2,540 |
| `pet_shop` | 0.0096 | 8.74% | 4.24 | -11.1 | 1,704 |
| `home_confort` | 0.0112 | 14.40% | 3.87 | -8.6 | 375 |
