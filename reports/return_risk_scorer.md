# Return-Risk Scorer (Secondary Analysis)

**Status:** Lightweight Exploratory Extension  
**Scope Gap Addressed:** Provides a dedicated return-risk proxy separate from the primary chargeback fraud model.

---

## 1. Known Limitations & Honesty Statement

> [!WARNING]
> **Not a Full ML Model:** This is a lightweight secondary analysis, not a full machine learning model with its own held-out test set, precision, or recall.
> 
> **Proxy Signal Used:** The Olist dataset does **not** contain an explicit `returned` flag. To build a defensible product-return risk scorer, we transparently constructed a **Return-Risk Proxy**.
> 
> A transaction is flagged as high return-risk if:
> 1. `order_status` == 'canceled' 
> 2. OR `review_score` == 1 (indicating severe dissatisfaction leading to refunds/returns)

This module is strictly additive and does not conflate general product returns with the primary Chargeback Evidence Responder's true fraud detection capabilities.

---

## 2. Correlation Analysis: Delivery Delay vs. Return Proxy

Does late delivery actually correlate with our return/cancellation proxy?
We calculated the Pearson correlation coefficient between the delivery delay (actual minus estimated delivery date in days) and the binary return proxy.

- **Correlation Coefficient ($r$):** `0.2513`
- **Finding:** A positive correlation confirms that delivery delays are a contributing signal to severe negative feedback and cancellations. 

---

## 3. Product Category Risk Rankings

Below are the product categories ranked by their historical proxy return rate (minimum 50 orders).

### Top 10 Highest Risk Categories

| Rank | Category | Total Orders | Return Proxy Rate | Avg Review Score | Avg Delay (Days) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | fashion_male_clothing | 112 | 23.21% | 3.70 | -12.73 |
| 2 | office_furniture | 1265 | 17.87% | 3.62 | -11.04 |
| 3 | construction_tools_safety | 162 | 17.28% | 3.89 | -12.21 |
| 4 | audio | 346 | 16.47% | 3.84 | -9.33 |
| 5 | unknown | 1437 | 16.21% | 3.92 | -10.64 |
| 6 | dvds_blu_ray | 59 | 15.25% | 4.09 | -12.46 |
| 7 | fixed_telephony | 217 | 15.21% | 3.90 | -14.24 |
| 8 | home_confort | 375 | 14.40% | 3.88 | -9.16 |
| 9 | air_conditioning | 252 | 13.10% | 4.04 | -13.20 |
| 10 | computers_accessories | 6660 | 12.79% | 4.02 | -11.66 |

### Top 10 Lowest Risk Categories (Safest)

| Rank | Category | Total Orders | Return Proxy Rate | Avg Review Score | Avg Delay (Days) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | food_drink | 224 | 4.91% | 4.38 | -10.66 |
| 2 | tablets_printing_image | 77 | 5.19% | 4.17 | -12.44 |
| 3 | construction_tools_lights | 233 | 6.44% | 4.19 | -10.60 |
| 4 | luggage_accessories | 1023 | 6.65% | 4.34 | -11.86 |
| 5 | books_general_interest | 509 | 7.07% | 4.47 | -11.20 |
| 6 | books_technical | 259 | 7.34% | 4.40 | -10.61 |
| 7 | costruction_tools_tools | 97 | 8.25% | 4.43 | -11.51 |
| 8 | stationery | 2294 | 8.50% | 4.25 | -11.30 |
| 9 | industry_commerce_and_business | 232 | 8.62% | 4.20 | -11.53 |
| 10 | pet_shop | 1704 | 8.74% | 4.24 | -11.65 |
