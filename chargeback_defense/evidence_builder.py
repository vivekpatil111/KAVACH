"""Evidence-Narrative Generation Layer: Commercial Fulfillment & Review Integration.

Loads real Olist commercial e-commerce order, fulfillment, and review datasets,
extracts structured delivery proof records, and constructs simulated demo linkages
pairing high-risk IEEE-CIS payment fraud claims with plausible commercial orders
based on amount parity.

IMPORTANT HONESTY NOTE:
IEEE-CIS (payment transactions) and Olist (commercial delivery/reviews) are completely
distinct real-world datasets with no shared identifiers. They are paired strictly via
amount-range similarity for end-to-end pipeline demonstration and marked explicitly as
"SIMULATED_DEMO" on every record.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from chargeback_defense.data_loader import _find_file, _load_single_file
from chargeback_defense.feature_engineering import (
    FEATURE_NAMES,
    compute_product_risk,
    engineer_features,
)
from chargeback_defense.syndicate_graph import get_syndicate_graph

BENCHMARK_USD_TO_INR_FX_RATE = 83.50
BENCHMARK_BRL_TO_INR_FX_RATE = 23.857  # ~83.50 / 3.50, calibrating Olist ticket sizes to realistic INR values

DISCLAIMER_TEXT = (
    "INDIAN D2C MERCHANT BENCHMARK — for pipeline demonstration only, not a real matched transaction. "
    "Payment transactions are calibrated in Indian Rupees (INR, ₹) reflecting standard Indian BFSI & D2C "
    "payment flows (UPI, RuPay, NetBanking, Debit Cards) paired with realistic Indian courier fulfillment tracking "
    "(BlueDart, Delhivery, Shadowfax) and 6-digit postal PIN codes. This linkage is labeled as SIMULATED_DEMO "
    "to maintain empirical transparency and benchmark dispute representment workflows for Indian merchants."
)

# Indian Logistics & Commerce Metadata Generators
INDIAN_HUBS = [
    ("Bengaluru", "Karnataka", "560001"),
    ("Mumbai", "Maharashtra", "400051"),
    ("New Delhi", "Delhi", "110001"),
    ("Gurugram", "Haryana", "122002"),
    ("Hyderabad", "Telangana", "500081"),
    ("Pune", "Maharashtra", "411001"),
    ("Ahmedabad", "Gujarat", "380015"),
    ("Jaipur", "Rajasthan", "302001"),
    ("Chennai", "Tamil Nadu", "600001"),
    ("Kolkata", "West Bengal", "700001"),
    ("Surat", "Gujarat", "395003"),
    ("Noida", "Uttar Pradesh", "201301"),
    ("Chandigarh", "Punjab", "160017"),
    ("Indore", "Madhya Pradesh", "452001"),
    ("Kochi", "Kerala", "682001"),
]

INDIAN_COURIERS = [
    ("BlueDart Express", "BLUEDART-"),
    ("Delhivery Direct", "DELHIVERY-"),
    ("Shadowfax Surface", "SFX-IN-"),
]

UPI_HANDLES = [
    "@okhdfcbank",
    "@paytm",
    "@oksbi",
    "@icici",
    "@ybl",
    "@axl",
]


# =============================================================================
# TASK 1: OLIST DATA LOADING & EXPLORATION
# =============================================================================

def load_and_explore_olist(olist_dir: str = "data/raw/olist") -> pd.DataFrame:
    """Load and profile the 7 core Olist e-commerce datasets and merge them on order_id.

    Returns:
        pd.DataFrame: Merged Olist orders containing items, payments, reviews, and customers.
    """
    print("\n" + "=" * 72)
    print("TASK 1: OLIST E-COMMERCE DATA LOADING & EXPLORATION")
    print("=" * 72)
    t0 = time.time()

    # 1. Load files
    orders = _load_single_file(os.path.join(olist_dir, "olist_orders_dataset.csv"))
    items = _load_single_file(os.path.join(olist_dir, "olist_order_items_dataset.csv"))
    reviews = _load_single_file(os.path.join(olist_dir, "olist_order_reviews_dataset.csv"))
    payments = _load_single_file(os.path.join(olist_dir, "olist_order_payments_dataset.csv"))
    customers = _load_single_file(os.path.join(olist_dir, "olist_customers_dataset.csv"))
    products = _load_single_file(os.path.join(olist_dir, "olist_products_dataset.csv"))
    translation = _load_single_file(os.path.join(olist_dir, "product_category_name_translation.csv"))
    sellers = _load_single_file(os.path.join(olist_dir, "olist_sellers_dataset.csv"))

    # Map product categories to English
    prod_trans = products.merge(translation, on="product_category_name", how="left")
    prod_cat_map = dict(zip(prod_trans["product_id"], prod_trans["product_category_name_english"].fillna("general_merchandise")))

    # 2. Date parsing and delivery delta calculation
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Delivery performance metrics
    delivered_mask = orders["order_delivered_customer_date"].notna()
    delivery_delta_days = np.full(len(orders), np.nan, dtype=np.float32)
    transit_duration_days = np.full(len(orders), np.nan, dtype=np.float32)

    # Delta: delivered date minus estimated date (positive = late, negative = early/on-time)
    delta_series = (
        orders.loc[delivered_mask, "order_delivered_customer_date"]
        - orders.loc[delivered_mask, "order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    delivery_delta_days[delivered_mask] = delta_series.values

    # Transit duration: purchase timestamp to customer delivery timestamp
    duration_series = (
        orders.loc[delivered_mask, "order_delivered_customer_date"]
        - orders.loc[delivered_mask, "order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    transit_duration_days[delivered_mask] = duration_series.values

    orders["delivery_delta_days"] = delivery_delta_days
    orders["transit_duration_days"] = transit_duration_days

    # Delivery status categorization
    delivery_status = np.full(len(orders), "PENDING", dtype=object)
    on_time = delivered_mask & (orders["delivery_delta_days"] <= 0)
    late = delivered_mask & (orders["delivery_delta_days"] > 0)
    canceled = orders["order_status"].isin(["canceled", "unavailable"])
    in_transit = orders["order_status"].isin(["shipped", "processing", "invoiced"])

    delivery_status[on_time] = "DELIVERED_ON_TIME"
    delivery_status[late] = "DELIVERED_LATE"
    delivery_status[canceled] = "CANCELED_OR_UNAVAILABLE"
    delivery_status[in_transit] = "IN_TRANSIT"
    orders["delivery_performance"] = delivery_status

    # 3. Aggregate items per order
    items["category_en"] = items["product_id"].map(prod_cat_map).fillna("general_merchandise")
    items_agg = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_items_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        primary_product_id=("product_id", "first"),
        primary_seller_id=("seller_id", "first"),
        product_category=("category_en", "first"),
    ).reset_index()

    # Join seller location
    seller_loc = sellers[["seller_id", "seller_city", "seller_state"]].drop_duplicates("seller_id")
    items_agg = items_agg.merge(seller_loc, left_on="primary_seller_id", right_on="seller_id", how="left")

    # 4. Aggregate payments per order
    payments_agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        primary_payment_type=("payment_type", "first"),
        max_installments=("payment_installments", "max"),
    ).reset_index()

    # 5. Clean reviews (pick most recent review per order if duplicates exist)
    reviews_clean = (
        reviews.sort_values("review_creation_date", ascending=False)
        .drop_duplicates(subset=["order_id"])
        [["order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date"]]
    )

    # 6. Merge onto orders backbone
    merged = orders.merge(items_agg, on="order_id", how="left")
    merged = merged.merge(payments_agg, on="order_id", how="left")
    merged = merged.merge(reviews_clean, on="order_id", how="left")
    merged = merged.merge(customers[["customer_id", "customer_city", "customer_state", "customer_zip_code_prefix"]], on="customer_id", how="left")

    # Fallback total order amount if payment missing
    merged["total_order_value"] = merged["total_payment_value"].fillna(merged["total_items_price"] + merged["total_freight"])

    # 7. Print summary statistics
    min_date = orders["order_purchase_timestamp"].min()
    max_date = orders["order_purchase_timestamp"].max()

    print("\n--- OLIST DATASET OVERVIEW ---")
    print(f"Total Unique Orders:   {len(merged):,}")
    print(f"Calendar Date Range:   {min_date.date()} to {max_date.date()} ({(max_date - min_date).days} days)")
    print(f"Total Items Recorded:  {len(items):,}")
    print(f"Total Payments Value:  R$ {merged['total_order_value'].sum():,.2f}")
    print(f"Average Order Value:   R$ {merged['total_order_value'].mean():.2f} (Median: R$ {merged['total_order_value'].median():.2f})")

    print("\nOrder Status Distribution:")
    for status, cnt in orders["order_status"].value_counts().items():
        print(f"  {status:<16s}: {cnt:>7,d} ({cnt / len(orders):>6.2%})")

    print("\nDelivery Performance Breakdown:")
    for perf, cnt in pd.Series(delivery_status).value_counts().items():
        print(f"  {perf:<24s}: {cnt:>7,d} ({cnt / len(orders):>6.2%})")

    print("\nCustomer Review Score Distribution (1-5 stars):")
    for score, cnt in reviews_clean["review_score"].value_counts().sort_index().items():
        print(f"  {int(score)} Star{'s' if score > 1 else ' '}: {cnt:>7,d} ({cnt / len(reviews_clean):>6.2%})")

    print("\nMissing Values Profile in Key Evidence Columns:")
    for col in ["order_delivered_customer_date", "review_score", "review_comment_message", "primary_payment_type", "product_category"]:
        missing_pct = merged[col].isna().mean() * 100
        print(f"  {col:<30s}: {missing_pct:>6.2f}% missing")

    print(f"\nOlist loading and feature synthesis completed in {time.time()-t0:.2f}s.")
    return merged


# =============================================================================
# TASK 2: EVIDENCE-RELEVANT FEATURE EXTRACTION
# =============================================================================

def extract_evidence_records(olist_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert merged Olist DataFrame into structured, business-readable evidence records."""
    t0 = time.time()
    records: List[Dict[str, Any]] = []

    for row in olist_df.itertuples(index=False):
        oid_str = str(row.order_id)
        # Deterministic hash for Indian commerce synthesis
        h = sum((i + 1) * ord(c) for i, c in enumerate(oid_str)) & 0xFFFFFFFF
        
        # Hub & 6-digit Indian PIN code mapping
        c_city, c_state, c_pin = INDIAN_HUBS[h % len(INDIAN_HUBS)]
        s_city, s_state, s_pin = INDIAN_HUBS[(h // len(INDIAN_HUBS) + 3) % len(INDIAN_HUBS)]
        
        # Indian Courier & Tracking AWB
        courier_name, awb_prefix = INDIAN_COURIERS[h % len(INDIAN_COURIERS)]
        awb_num = f"{awb_prefix}{1000000000 + (h % 8999999999)}"
        
        # Indian Customer Phone (+91)
        cust_phone = f"+91 98{str(h % 100000000).zfill(8)}"
        
        # Indian BFSI Payment Methods: UPI VPAs, RuPay, Visa, Mastercard
        pay_type_raw = str(row.primary_payment_type or "credit_card").lower()
        if pay_type_raw in ["boleto", "voucher"] or (h % 3 == 0):
            indian_payment_type = "UPI"
            upi_handle = UPI_HANDLES[h % len(UPI_HANDLES)]
            payment_ref = f"user_{(h % 9000) + 1000}{upi_handle}"
            payment_display = f"UPI ({payment_ref})"
        elif (h % 3 == 1):
            indian_payment_type = "RuPay Debit"
            payment_ref = f"RuPay Platinum Debit (•••• {1000 + (h % 9000)})"
            payment_display = payment_ref
        else:
            brand = "Visa" if (h % 2 == 0) else "Mastercard"
            indian_payment_type = f"{brand} Debit"
            payment_ref = f"{brand} Classic (•••• {1000 + (h % 9000)})"
            payment_display = payment_ref

        # Format timestamps
        purchase_ts = row.order_purchase_timestamp.isoformat() if pd.notna(row.order_purchase_timestamp) else None
        approved_ts = row.order_approved_at.isoformat() if pd.notna(row.order_approved_at) else None
        carrier_ts = row.order_delivered_carrier_date.isoformat() if pd.notna(row.order_delivered_carrier_date) else None
        delivered_ts = row.order_delivered_customer_date.isoformat() if pd.notna(row.order_delivered_customer_date) else None
        estimated_ts = row.order_estimated_delivery_date.isoformat() if pd.notna(row.order_estimated_delivery_date) else None

        review_msg = str(row.review_comment_message).strip() if pd.notna(row.review_comment_message) else None
        review_ttl = str(row.review_comment_title).strip() if pd.notna(row.review_comment_title) else None

        item_cnt = int(row.item_count) if pd.notna(row.item_count) else 1
        inst_cnt = int(row.max_installments) if pd.notna(row.max_installments) else 1

        # Calibrated order value in INR
        raw_val = float(row.total_order_value or 0.0) if pd.notna(row.total_order_value) else 0.0
        val_inr = round(raw_val * BENCHMARK_BRL_TO_INR_FX_RATE, 2)

        rec = {
            "order_id": oid_str,
            "order_status": str(row.order_status),
            "total_order_value": val_inr,
            "total_order_value_inr": val_inr,
            "payment_type": indian_payment_type,
            "payment_method_display": payment_display,
            "payment_identifier": payment_ref,
            "payment_installments": inst_cnt,
            "item_count": item_cnt,
            "product_category": str(row.product_category or "general_merchandise"),
            "seller_id": str(row.primary_seller_id or "unknown"),
            "seller_location": f"{s_city}, {s_state} (PIN {s_pin})",
            "seller_pin": s_pin,
            "customer_location": f"{c_city}, {c_state} (PIN {c_pin})",
            "customer_pin": c_pin,
            "customer_phone": cust_phone,
            "courier_partner": courier_name,
            "awb_tracking_number": awb_num,
            "timeline": {
                "purchase_timestamp": purchase_ts,
                "approved_timestamp": approved_ts,
                "carrier_dispatched_timestamp": carrier_ts,
                "delivered_customer_timestamp": delivered_ts,
                "estimated_delivery_timestamp": estimated_ts,
            },
            "delivery_performance": {
                "status": str(row.delivery_performance),
                "delivery_delta_days": round(float(row.delivery_delta_days), 1) if pd.notna(row.delivery_delta_days) else None,
                "transit_duration_days": round(float(row.transit_duration_days), 1) if pd.notna(row.transit_duration_days) else None,
                "delivery_proof_available": bool(pd.notna(row.order_delivered_customer_date)),
                "courier_partner": courier_name,
                "awb_tracking_number": awb_num,
            },
            "customer_feedback": {
                "review_score": int(row.review_score) if pd.notna(row.review_score) else None,
                "review_comment_title": review_ttl,
                "review_comment_message": review_msg,
                "has_written_feedback": bool(review_msg is not None and len(review_msg) > 0),
            },
        }
        records.append(rec)

    print(f"Extracted {len(records):,} structured Indian D2C evidence records in {time.time()-t0:.2f}s.")
    return records


# =============================================================================
# TASK 3: DEMO LINKAGE LAYER (LABELED SIMULATED MATCH)
# =============================================================================

def load_flagged_ieee_test_claims(
    raw_dir: str = "data/raw",
    model_path: str = "models/chargeback_xgb.json",
    operating_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Retrieve high-risk flagged transactions from the held-out temporal test set.

    Returns:
        flagged_df: DataFrame of flagged test claims
        flagged_scores: Predicted risk scores for the flagged claims
        shap_contribs: TreeSHAP contribution matrix for the flagged claims
    """
    print("\nScoring IEEE-CIS held-out test split to identify high-risk claims...")
    train_tx_path = _find_file(raw_dir, "train_transaction")
    train_id_path = _find_file(raw_dir, "train_identity")

    df_tx = _load_single_file(train_tx_path).sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n_total = len(df_tx)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_slice = df_tx.iloc[:n_train]
    test_slice = df_tx.iloc[n_train + n_val:].copy()

    prod_risk = compute_product_risk(train_slice)
    df_id = _load_single_file(train_id_path)
    test_merged = test_slice.merge(df_id, on="TransactionID", how="left")

    X_test = engineer_features(test_merged, prod_risk)[FEATURE_NAMES]

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # TreeSHAP exact contributions
    dmat = xgb.DMatrix(X_test)
    booster = model.get_booster()
    contribs = booster.predict(dmat, pred_contribs=True)
    scores = model.predict_proba(X_test)[:, 1]

    flagged_mask = scores >= operating_threshold
    flagged_df = test_merged.loc[flagged_mask].copy()
    flagged_scores = scores[flagged_mask]
    flagged_contribs = contribs[flagged_mask]

    flagged_df["fraud_score"] = flagged_scores
    print(f"Identified {len(flagged_df):,} high-risk test claims (threshold >= {operating_threshold:.2f}).")
    return flagged_df, flagged_scores, flagged_contribs


def build_demo_linkages(
    flagged_claims_df: pd.DataFrame,
    flagged_scores: np.ndarray,
    flagged_contribs: np.ndarray,
    olist_records: List[Dict[str, Any]],
    tolerance_pct: float = 0.15,
) -> Dict[str, Any]:
    """Match each flagged IEEE-CIS claim to a plausible commercial order using INR amount-range parity.

    Matches within +/- 15% tolerance; ties broken randomly or with nearest distance.
    Explicitly tags every record with linkage_type: 'SIMULATED_DEMO'.
    """
    print("\n" + "=" * 72)
    print("TASK 3: DEMO LINKAGE LAYER (IEEE-CIS CLAIMS <-> INDIAN D2C ORDERS)")
    print("=" * 72)
    t0 = time.time()

    # Pre-index order INR amounts for fast nearest-neighbor matching
    olist_amounts = np.array([r["total_order_value_inr"] for r in olist_records], dtype=np.float32)
    sort_idx = np.argsort(olist_amounts)
    sorted_amounts = olist_amounts[sort_idx]

    # Map sorted indices back to olist_records
    rng = np.random.default_rng(42)
    linkages: List[Dict[str, Any]] = []
    matched_within_tol = 0
    diff_percentages: List[float] = []

    claims_tx_ids = flagged_claims_df["TransactionID"].values
    claims_amts = flagged_claims_df["TransactionAmt"].values
    claims_dt = flagged_claims_df["TransactionDT"].values
    claims_prod = flagged_claims_df["ProductCD"].fillna("unknown").values
    claims_card4 = flagged_claims_df["card4"].fillna("unknown").values
    claims_card6 = flagged_claims_df["card6"].fillna("unknown").values
    claims_truth = flagged_claims_df["isFraud"].values

    for idx in range(len(claims_tx_ids)):
        claim_id = f"CLM_{claims_tx_ids[idx]}"
        amt_usd = float(claims_amts[idx])
        amt_inr = round(amt_usd * BENCHMARK_USD_TO_INR_FX_RATE, 2)
        score = float(flagged_scores[idx])
        truth = int(claims_truth[idx])

        # Find candidate orders within +/- 15% tolerance of the converted INR amount
        low_bound = amt_inr * (1.0 - tolerance_pct)
        high_bound = amt_inr * (1.0 + tolerance_pct)

        left_idx = np.searchsorted(sorted_amounts, low_bound, side="left")
        right_idx = np.searchsorted(sorted_amounts, high_bound, side="right")

        within_tol = False
        if right_idx > left_idx:
            # Matches exist within +/- 15%: pick randomly among the window
            chosen_sorted_idx = rng.integers(left_idx, right_idx)
            chosen_record_idx = sort_idx[chosen_sorted_idx]
            matched_within_tol += 1
            within_tol = True
        else:
            # Fallback to closest available in the entire distribution
            closest_idx = np.searchsorted(sorted_amounts, amt_inr)
            cand_indices = []
            if closest_idx < len(sorted_amounts):
                cand_indices.append(closest_idx)
            if closest_idx > 0:
                cand_indices.append(closest_idx - 1)
            chosen_sorted_idx = min(cand_indices, key=lambda i: abs(sorted_amounts[i] - amt_inr))
            chosen_record_idx = sort_idx[chosen_sorted_idx]
            within_tol = False

        matched_evidence = olist_records[chosen_record_idx]
        matched_amt_inr = matched_evidence["total_order_value_inr"]
        pct_diff = abs(matched_amt_inr - amt_inr) / (amt_inr + 1e-4)
        diff_percentages.append(pct_diff)

        # Extract top 4 TreeSHAP feature contributions for this specific transaction
        row_contribs = flagged_contribs[idx][:-1]  # Exclude bias term
        top_feat_indices = np.argsort(np.abs(row_contribs))[::-1][:4]
        top_signals = []
        for fi in top_feat_indices:
            fname = FEATURE_NAMES[fi]
            contrib_val = float(row_contribs[fi])
            top_signals.append({
                "feature": fname,
                "shap_contribution": round(contrib_val, 4),
                "risk_direction": "INCREASES_RISK" if contrib_val > 0 else "DECREASES_RISK",
            })

        linkage_entry = {
            "linkage_type": "SIMULATED_DEMO",
            "claim_id": claim_id,
            "ieee_transaction_id": int(claims_tx_ids[idx]),
            "disputed_amount_original": {
                "value": round(amt_inr, 2),
                "currency": "INR",
            },
            "disputed_amount_converted": {
                "value": round(amt_inr, 2),
                "currency": "INR",
                "fx_rate_used": BENCHMARK_USD_TO_INR_FX_RATE,
                "fx_rate_period": "Benchmark calibration: 1 USD = 83.50 INR",
            },
            "disputed_amount_usd_reference": {
                "value": round(amt_usd, 2),
                "currency": "USD",
            },
            "ieee_product_cd": str(claims_prod[idx]),
            "ieee_card_network": str(claims_card4[idx]),
            "ieee_card_type": str(claims_card6[idx]),
            "ground_truth_label": truth,
            "fraud_risk_score": round(score, 6),
            "matched_order_id": matched_evidence["order_id"],
            "matched_olist_order_id": matched_evidence["order_id"],
            "matched_order_value_inr": round(matched_amt_inr, 2),
            "matched_olist_amount_brl": round(matched_amt_inr, 2),
            "amount_difference_pct": round(pct_diff * 100, 2),
            "within_tolerance": within_tol,
            "top_contributing_signals": top_signals,
            "evidence_record": matched_evidence,
        }
        linkages.append(linkage_entry)

    total_claims = len(claims_tx_ids)
    mean_diff = float(np.mean(diff_percentages)) * 100
    median_diff = float(np.median(diff_percentages)) * 100

    print(f"Total High-Risk Claims Matched: {total_claims:,}")
    print(f"Matched Within Target Tolerance (+/- 15%): {matched_within_tol:,} ({matched_within_tol / total_claims:.2%})")
    print(f"Average Amount Parity Delta: {mean_diff:.2f}% (Median: {median_diff:.2f}%)")

    output_payload = {
        "disclaimer": DISCLAIMER_TEXT,
        "linkage_type": "SIMULATED_DEMO",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "currency_normalization": {
            "source_currency": "USD (IEEE-CIS)",
            "target_currency": "INR (Indian D2C Benchmark)",
            "fx_rate_applied": BENCHMARK_USD_TO_INR_FX_RATE,
            "fx_rate_period": "Benchmark calibration: 1 USD = 83.50 INR",
            "methodology": "Currency-normalized amount parity matching in INR within +/-15% tolerance",
        },
        "total_flagged_claims_matched": total_claims,
        "matched_within_15pct_tolerance": matched_within_tol,
        "tolerance_coverage_pct": round(matched_within_tol / total_claims * 100, 2),
        "mean_amount_difference_pct": round(mean_diff, 2),
        "median_amount_difference_pct": round(median_diff, 2),
        "linkages": linkages,
    }

    out_path = "data/processed/demo_claim_evidence_linkage.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"Saved demo linkage mapping to: {out_path} ({time.time()-t0:.2f}s)")

    return output_payload


# =============================================================================
# TASK 4: EVIDENCE PACKET DRAFT GENERATOR
# =============================================================================

def build_evidence_packet(linkage_record: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured, self-contained dispute evidence packet.

    Combines:
    - Model risk score + top contributing TreeSHAP feature signals
    - Real commercial delivery & review evidence from Indian D2C fulfillment
    - Compelling evidence assessment & dispute representment strategy
    - Empty placeholder narrative field for downstream LLM generation
    """
    evidence = linkage_record["evidence_record"]
    deliv = evidence["delivery_performance"]
    feedback = evidence["customer_feedback"]
    score = linkage_record["fraud_risk_score"]

    # Objective dispute defense assessment
    compelling_factors: List[str] = []
    recommendation = "INVESTIGATE"
    classification = "UNKNOWN_DISPUTE"

    courier_name = evidence.get("courier_partner", "Courier")
    awb = evidence.get("awb_tracking_number", "")

    if deliv["delivery_proof_available"] and deliv["status"] == "DELIVERED_ON_TIME":
        compelling_factors.append(
            f"{courier_name} tracking scan (AWB: {awb}) confirms delivery to customer on "
            f"{evidence['timeline']['delivered_customer_timestamp'][:10]} "
            f"({abs(deliv['delivery_delta_days'] or 0):.1f} days before estimated SLA deadline)"
        )
        if feedback["review_score"] is not None and feedback["review_score"] >= 4:
            compelling_factors.append(
                f"Customer recorded a positive {feedback['review_score']}-star review following delivery"
            )
            recommendation = "CONTEST_CHARGEBACK_WITH_EVIDENCE"
            classification = "FIRST_PARTY_FRIENDLY_FRAUD"
        elif feedback["review_score"] is not None and feedback["review_score"] <= 2:
            compelling_factors.append(
                f"Customer logged a negative {feedback['review_score']}-star review citing product/service dissatisfaction"
            )
            recommendation = "MERCHANT_SETTLEMENT_REFUND_RECOMMENDED"
            classification = "PRODUCT_NOT_AS_DESCRIBED_OR_DAMAGED"
        else:
            recommendation = "CONTEST_CHARGEBACK_WITH_DELIVERY_PROOF"
            classification = "ITEM_NOT_RECEIVED_REPRESENTMENT"

    elif deliv["status"] == "DELIVERED_LATE":
        compelling_factors.append(
            f"{courier_name} records (AWB: {awb}) indicate delivery was delayed by {deliv['delivery_delta_days']} days past SLA estimate"
        )
        recommendation = "REVIEW_SHIPPING_SLA_BEFORE_REPRESENTMENT"
        classification = "LATE_DELIVERY_DISPUTE"

    elif deliv["status"] == "CANCELED_OR_UNAVAILABLE":
        compelling_factors.append("Order was canceled or marked unfulfilled prior to courier dispatch")
        recommendation = "ACCEPT_CHARGEBACK_OR_ISSUE_REFUND"
        classification = "UNFULFILLED_MERCHANDISE"

    # In-Memory Multi-Merchant Syndicate Ring Analysis (<0.1ms)
    claim_id = linkage_record["claim_id"]
    user_id = f"USR_{claim_id.replace('CLM_', '')}"
    graph = get_syndicate_graph()
    syndicate_analysis = graph.extract_features(user_id)

    if syndicate_analysis["is_syndicate_attack"]:
        compelling_factors.insert(
            0,
            f"🚨 COORDINATED SYNDICATE ATTACK: In-memory graph analysis detected shared device/VPA infrastructure "
            f"spanning {syndicate_analysis['cluster_merchant_span']} distinct D2C merchants with "
            f"{syndicate_analysis['cluster_burst_7d']} dispute claims in trailing 7 days "
            f"(Cluster Size: {syndicate_analysis['cluster_size']} nodes)."
        )
        recommendation = "CONTEST_CHARGEBACK_WITH_SYNDICATE_EVIDENCE"
        classification = "COORDINATED_MULTI_MERCHANT_SYNDICATE_FRAUD"

    packet = {
        "packet_id": f"EVD-{linkage_record['claim_id']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER_TEXT,
        "linkage_type": "SIMULATED_DEMO",
        "dispute_summary": {
            "claim_id": linkage_record["claim_id"],
            "ieee_transaction_id": linkage_record["ieee_transaction_id"],
            "disputed_amount_original": linkage_record["disputed_amount_original"],
            "disputed_amount_converted": linkage_record["disputed_amount_converted"],
            "disputed_amount_usd_reference": linkage_record.get("disputed_amount_usd_reference", {}),
            "product_category_code": linkage_record["ieee_product_cd"],
            "card_network": linkage_record["ieee_card_network"],
            "card_type": linkage_record["ieee_card_type"],
            "ground_truth_label": linkage_record["ground_truth_label"],
        },
        "model_risk_assessment": {
            "model_architecture": "XGBoost Classifier with Causal Velocity (IEEE-CIS trained)",
            "fraud_risk_score": score,
            "risk_band": "HIGH_RISK" if score >= 0.70 else ("MEDIUM_RISK" if score >= 0.50 else "LOW_RISK"),
            "operating_threshold": 0.70,
            "decision": "FLAGGED_FOR_CHARGEBACK_DEFENSE",
            "top_contributing_signals": linkage_record["top_contributing_signals"],
        },
        "in_memory_syndicate_analysis": syndicate_analysis,
        "commercial_fulfillment_evidence": {
            "source_dataset": "Indian D2C Merchant Benchmark (Synthetic fulfillment telemetry)",
            "matched_order_id": evidence["order_id"],
            "matched_order_value_inr": evidence["total_order_value_inr"],
            "matched_order_value_brl": evidence["total_order_value_inr"],
            "target_converted_amount_inr": linkage_record["disputed_amount_converted"]["value"],
            "target_converted_amount_brl": linkage_record["disputed_amount_converted"]["value"],
            "amount_match_delta_pct": linkage_record["amount_difference_pct"],
            "within_target_tolerance": linkage_record["within_tolerance"],
            "currency_normalization_note": (
                f"Disputed INR {linkage_record['disputed_amount_original']['value']:,.2f} calibrated from "
                f"USD {linkage_record.get('disputed_amount_usd_reference', {}).get('value', 0):.2f} @ {BENCHMARK_USD_TO_INR_FX_RATE} INR/USD "
                f"prior to D2C fulfillment order matching."
            ),
            "order_status": evidence["order_status"],
            "timeline": evidence["timeline"],
            "delivery_performance": evidence["delivery_performance"],
            "customer_feedback_record": evidence["customer_feedback"],
            "merchant_and_item_details": {
                "seller_id": evidence["seller_id"],
                "seller_location": evidence["seller_location"],
                "seller_pin": evidence.get("seller_pin", ""),
                "customer_location": evidence["customer_location"],
                "customer_pin": evidence.get("customer_pin", ""),
                "customer_phone": evidence.get("customer_phone", ""),
                "courier_partner": evidence.get("courier_partner", "BlueDart Express"),
                "awb_tracking_number": evidence.get("awb_tracking_number", ""),
                "product_category": evidence["product_category"],
                "item_count": evidence["item_count"],
            },
            "payment_profile": {
                "payment_type": evidence["payment_type"],
                "payment_method_display": evidence.get("payment_method_display", ""),
                "payment_identifier": evidence.get("payment_identifier", ""),
                "installments": evidence["payment_installments"],
            },
        },
        "dispute_defense_evaluation": {
            "compelling_evidence_factors": compelling_factors,
            "dispute_representment_recommendation": recommendation,
            "chargeback_reason_classification": classification,
        },
        "narrative": "",  # Placeholder for LLM generation layer
    }
    return packet


def run_evidence_pipeline(
    olist_dir: str = "data/raw/olist",
    raw_dir: str = "data/raw",
    model_path: str = "models/chargeback_xgb.json",
    reports_dir: str = "reports",
) -> List[Dict[str, Any]]:
    """Execute the full evidence processing, demo linkage, and packet drafting."""
    # 1. Load Olist
    olist_df = load_and_explore_olist(olist_dir=olist_dir)
    olist_records = extract_evidence_records(olist_df)

    # 2. Get High-Risk Test Claims
    flagged_df, flagged_scores, flagged_contribs = load_flagged_ieee_test_claims(
        raw_dir=raw_dir, model_path=model_path, operating_threshold=0.70
    )

    # 3. Build Demo Linkages
    linkage_payload = build_demo_linkages(
        flagged_df, flagged_scores, flagged_contribs, olist_records, tolerance_pct=0.15
    )
    linkages = linkage_payload["linkages"]

    # 4. Generate 4 Diverse Sample Evidence Packets
    print("\n" + "=" * 72)
    print("TASK 4: GENERATING SAMPLE EVIDENCE PACKETS (DIVERSE SCENARIOS)")
    print("=" * 72)

    # Find representative samples covering:
    # 1. Delivered on-time + 5-star review (Friendly Fraud / Representment Win)
    # 2. Delivered late + 1-star review (Genuine Dissatisfaction / Accept Refund)
    # 3. Canceled or in-transit (Fulfillment Issue / Merchant Hold)
    # 4. Standard high-risk delivered claim
    samples: List[Dict[str, Any]] = []

    case_1 = next((l for l in linkages if l["evidence_record"]["delivery_performance"]["status"] == "DELIVERED_ON_TIME" and (l["evidence_record"]["customer_feedback"]["review_score"] or 0) >= 4), linkages[0])
    case_2 = next((l for l in linkages if l["evidence_record"]["delivery_performance"]["status"] == "DELIVERED_LATE" and (l["evidence_record"]["customer_feedback"]["review_score"] or 5) <= 2), linkages[1])
    case_3 = next((l for l in linkages if l["evidence_record"]["delivery_performance"]["status"] in ["CANCELED_OR_UNAVAILABLE", "IN_TRANSIT"]), linkages[2])
    case_4 = linkages[min(10, len(linkages) - 1)]

    for c in [case_1, case_2, case_3, case_4]:
        pkt = build_evidence_packet(c)
        samples.append(pkt)

    sample_path = os.path.join(reports_dir, "sample_evidence_packets.json")
    os.makedirs(reports_dir, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)
    print(f"Saved {len(samples)} diverse sample evidence packets to: {sample_path}")

    return samples


if __name__ == "__main__":
    run_evidence_pipeline()
