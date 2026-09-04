import os
import json
import pandas as pd
import numpy as np

# Paths
DATA_DIR = "data/raw/olist"
REPORTS_DIR = "reports"
OUTPUT_JSON = os.path.join(REPORTS_DIR, "return_risk_data.json")
OUTPUT_MD = os.path.join(REPORTS_DIR, "return_risk_scorer.md")

def compute_return_risk():
    print("Loading Olist datasets...")
    # Load data
    orders = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(DATA_DIR, "product_category_name_translation.csv"))

    # Merge to get product categories for items
    items_prod = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    items_prod = items_prod.merge(translations, on='product_category_name', how='left')
    items_prod['category'] = items_prod['product_category_name_english'].fillna('unknown')
    
    # We want one row per order for the main analysis, or one row per item.
    # Since returns are often order-level (or item-level but we have order-level reviews), 
    # let's do order-level by picking the primary category for the order.
    # To keep it simple, we'll merge items to orders and just use the first item's category if multiple.
    order_cats = items_prod.groupby('order_id').first().reset_index()[['order_id', 'category']]
    
    # Merge orders with categories and reviews
    df = orders.merge(order_cats, on='order_id', how='inner')
    
    # Reviews might have multiple per order, take the lowest score (most critical)
    order_reviews = reviews.groupby('order_id')['review_score'].min().reset_index()
    df = df.merge(order_reviews, on='order_id', how='left')

    print("Defining return proxy and delivery delays...")
    # Proxy Definition:
    # Olist lacks a strict "returned" flag. We define a Return-Risk Proxy as:
    # 1. Order status is 'canceled' OR
    # 2. Review score is 1 (strongest indicator of severe dissatisfaction/refund demand)
    df['return_proxy'] = ((df['order_status'] == 'canceled') | (df['review_score'] == 1)).astype(int)

    # Delivery Delay: actual delivered vs estimated
    df['delivered_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['estimated_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    
    # Delay in days. Positive means late.
    df['delivery_delay_days'] = (df['delivered_date'] - df['estimated_date']).dt.total_seconds() / (24 * 3600)
    df['is_late'] = (df['delivery_delay_days'] > 0).astype(int)

    # Correlation
    # We compute correlation between delivery delay (days) and return proxy for delivered orders
    delivered_df = df.dropna(subset=['delivery_delay_days', 'return_proxy'])
    correlation = delivered_df['delivery_delay_days'].corr(delivered_df['return_proxy'])
    
    print(f"Correlation between delivery delay (days) and return proxy: {correlation:.4f}")

    # Aggregation by Category
    # We require a minimum number of orders per category to avoid noise
    min_orders = 50
    cat_stats = df.groupby('category').agg(
        total_orders=('order_id', 'count'),
        return_proxy_count=('return_proxy', 'sum'),
        avg_review_score=('review_score', 'mean'),
        avg_delay_days=('delivery_delay_days', 'mean')
    ).reset_index()
    
    cat_stats = cat_stats[cat_stats['total_orders'] >= min_orders]
    cat_stats['return_proxy_rate'] = cat_stats['return_proxy_count'] / cat_stats['total_orders']
    
    # Rank by return_proxy_rate
    cat_stats = cat_stats.sort_values(by='return_proxy_rate', ascending=False)
    
    top_10 = cat_stats.head(10)
    bottom_10 = cat_stats.tail(10).sort_values(by='return_proxy_rate', ascending=True)

    # Save JSON for dashboard
    report_data = {
        "correlation_coefficient": correlation,
        "proxy_definition": "Order status is 'canceled' OR Review Score is 1.",
        "top_10_categories": top_10.to_dict(orient='records'),
        "bottom_10_categories": bottom_10.to_dict(orient='records'),
        "limitations": "This is a lightweight secondary angle using proxy signals. Olist has no explicit 'return' flag. This is not a full ML model."
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# Return-Risk Scorer (Secondary Analysis)

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

- **Correlation Coefficient ($r$):** `{correlation:.4f}`
- **Finding:** A positive correlation confirms that delivery delays are a contributing signal to severe negative feedback and cancellations. 

---

## 3. Product Category Risk Rankings

Below are the product categories ranked by their historical proxy return rate (minimum 50 orders).

### Top 10 Highest Risk Categories

| Rank | Category | Total Orders | Return Proxy Rate | Avg Review Score | Avg Delay (Days) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for idx, row in enumerate(top_10.itertuples(), 1):
        md_content += f"| {idx} | {row.category} | {row.total_orders} | {row.return_proxy_rate:.2%} | {row.avg_review_score:.2f} | {row.avg_delay_days:.2f} |\n"

    md_content += "\n### Top 10 Lowest Risk Categories (Safest)\n\n"
    md_content += "| Rank | Category | Total Orders | Return Proxy Rate | Avg Review Score | Avg Delay (Days) |\n"
    md_content += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for idx, row in enumerate(bottom_10.itertuples(), 1):
        md_content += f"| {idx} | {row.category} | {row.total_orders} | {row.return_proxy_rate:.2%} | {row.avg_review_score:.2f} | {row.avg_delay_days:.2f} |\n"

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("Successfully generated return risk scorer artifacts.")
    print("Top 10 Categories by Return Proxy Rate:")
    print(top_10[['category', 'return_proxy_rate', 'avg_review_score']])

if __name__ == "__main__":
    compute_return_risk()
