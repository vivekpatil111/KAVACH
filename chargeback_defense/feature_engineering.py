"""First-principles feature engineering for Chargeback Defense & Return-Risk Scoring.

Designed specifically for chargeback evidence preparation and return risk:
- Causal trailing velocity without lookahead leakage
- Card-billing-shipping mismatch detection
- ProductCD risk encoding (strictly from train split)
- Amount anomaly & z-score relative to card history
- Device & identity presence consistency
- Address Verification Service (AVS) match signals
"""

from __future__ import annotations

from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Constants for trailing time windows (TransactionDT is in seconds)
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 86400
WINDOW_24H = 86400
WINDOW_7D = 7 * 86400

# Feature names explicitly registered for explainability
FEATURE_NAMES: List[str] = [
    # 1. Transaction Velocity
    "card_txn_count_24h",
    "card_txn_count_7d",
    "card_txn_amt_sum_24h",
    "card_time_since_last_txn_hours",
    "email_domain_txn_count_24h",
    # 2. Amount Anomaly & Relative Spending
    "card_amt_zscore",
    "amt_to_card_mean_ratio",
    "amt_log",
    "amt_is_round_dollar",
    "amt_cents",
    # 3. Product Category Risk
    "ProductCD_train_fraud_rate",
    # 4. Card-Billing-Shipping Mismatches
    "addr_card_country_mismatch",
    "has_billing_addr",
    "has_shipping_dist",
    "shipping_billing_dist",
    "email_domain_mismatch",
    "has_recipient_email",
    # 5. Device & Identity Consistency
    "has_identity_data",
    "is_mobile_device",
    "is_desktop_device",
    "has_device_info",
    "browser_is_known",
    "os_is_known",
    "ip_proxy_risk_flag",
    # 6. AVS & Verification Matches
    "avs_address_match",
    "avs_billing_match",
    "avs_shipping_match",
    "total_verification_matches",
    # 7. Card & Historical Context
    "card_address_count_C1",
    "transaction_count_C2",
    "email_count_C13",
    "time_delta_card_creation_D1",
    "time_delta_prev_txn_D2",
    # 8. Temporal Context
    "hour_of_day",
    "is_night_transaction",
    "day_of_week",
]


def compute_product_risk(train_df: pd.DataFrame) -> Dict[str, float]:
    """Compute empirical Bayes smoothed fraud rate per ProductCD on the train split only.
    
    Formula: (fraud_count + m * prior) / (total_count + m)
    where prior = 0.035, m = 10.
    """
    prior = 0.035
    m = 10.0
    stats = train_df.groupby("ProductCD")["isFraud"].agg(["sum", "count"])
    risk_map: Dict[str, float] = {}
    for prod_cd, row in stats.iterrows():
        smoothed = (row["sum"] + m * prior) / (row["count"] + m)
        risk_map[str(prod_cd)] = float(smoothed)
    return risk_map


def build_card_key(df: pd.DataFrame) -> np.ndarray:
    """Construct composite card identifier: card1_card2_card3_card4_card6."""
    c1 = df["card1"].fillna(-1).astype(int).astype(str)
    c2 = df["card2"].fillna(-1).astype(int).astype(str)
    c3 = df["card3"].fillna(-1).astype(int).astype(str)
    c4 = df["card4"].fillna("unknown").astype(str)
    c6 = df["card6"].fillna("unknown").astype(str)
    return (c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c6).values


def engineer_features(
    df: pd.DataFrame,
    product_risk_map: Dict[str, float],
) -> pd.DataFrame:
    """Compute all 35 domain-grounded features on a chronologically sorted DataFrame.
    
    All velocity and expanding statistics are computed strictly causally: for transaction i,
    only transactions j < i with TransactionDT_j <= TransactionDT_i are visible.
    
    Args:
        df: DataFrame sorted chronologically by TransactionDT.
        product_risk_map: Pre-computed fraud rates per ProductCD from the train split.
        
    Returns:
        pd.DataFrame containing FEATURE_NAMES columns.
    """
    t0 = time.time()
    n = len(df)
    times = df["TransactionDT"].values
    amts = df["TransactionAmt"].values.astype(np.float32)
    card_keys = build_card_key(df)
    p_emails = df["P_emaildomain"].fillna("").astype(str).values

    # Pre-allocate output arrays for speed
    card_count_24h = np.zeros(n, dtype=np.int32)
    card_count_7d = np.zeros(n, dtype=np.int32)
    card_amt_sum_24h = np.zeros(n, dtype=np.float32)
    card_time_since_last = np.full(n, -1.0, dtype=np.float32)
    email_count_24h = np.zeros(n, dtype=np.int32)

    card_amt_zscore = np.zeros(n, dtype=np.float32)
    amt_to_card_mean_ratio = np.ones(n, dtype=np.float32)

    # 1. Causal Loop for Trailing Velocity and Historical Card Spending
    # card_history: card_key -> deque of (timestamp, amount)
    card_history: Dict[str, deque] = defaultdict(deque)
    email_history: Dict[str, deque] = defaultdict(deque)
    last_card_time: Dict[str, float] = {}

    # card_stats: card_key -> [count, sum, sum_sq]
    card_stats: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])

    for i in range(n):
        t = times[i]
        a = amts[i]
        c = card_keys[i]
        em = p_emails[i]

        # Trailing 24h & 7d Card Velocity
        dq = card_history[c]
        # Prune transactions older than 7 days from the deque
        while dq and t - dq[0][0] > WINDOW_7D:
            dq.popleft()

        # Count 7d
        card_count_7d[i] = len(dq)

        # Count & Sum 24h (iterate backwards or slice)
        c24 = 0
        s24 = 0.0
        for past_t, past_a in reversed(dq):
            if t - past_t <= WINDOW_24H:
                c24 += 1
                s24 += past_a
            else:
                break
        card_count_24h[i] = c24
        card_amt_sum_24h[i] = s24

        # Inter-transaction duration
        if c in last_card_time:
            card_time_since_last[i] = (t - last_card_time[c]) / SECONDS_IN_HOUR
        last_card_time[c] = t

        # Trailing 24h Email Domain Velocity
        if em:
            edq = email_history[em]
            while edq and t - edq[0] > WINDOW_24H:
                edq.popleft()
            email_count_24h[i] = len(edq)
            edq.append(t)

        # Append current card transaction to history for FUTURE transactions
        dq.append((t, a))

        # Amount Z-score relative to past card transactions
        st = card_stats[c]
        prior_cnt = st[0]
        if prior_cnt > 0:
            mean = st[1] / prior_cnt
            var = max(0.0, (st[2] / prior_cnt) - (mean * mean))
            std = np.sqrt(var)
            card_amt_zscore[i] = (a - mean) / (std + 1.0)
            amt_to_card_mean_ratio[i] = a / (mean + 1e-4)
        else:
            card_amt_zscore[i] = 0.0
            amt_to_card_mean_ratio[i] = 1.0

        # Update card stats for FUTURE transactions
        st[0] += 1
        st[1] += a
        st[2] += a * a

    # 2. Vectorized Amount Features
    amt_log = np.log1p(amts)
    amt_is_round = (np.abs(amts - np.round(amts)) < 1e-4).astype(np.int32)
    amt_cents = (amts % 1.0).astype(np.float32)

    # 3. Product Category Risk Mapping
    prod_risk = df["ProductCD"].map(product_risk_map).fillna(0.035).values.astype(np.float32)

    # 4. Card-Billing-Shipping Mismatches
    card3_vals = df["card3"].values
    addr2_vals = df["addr2"].values
    both_country = (~np.isnan(card3_vals)) & (~np.isnan(addr2_vals))
    country_mismatch = np.zeros(n, dtype=np.int32)
    country_mismatch[both_country] = (card3_vals[both_country] != addr2_vals[both_country]).astype(np.int32)

    has_billing_addr = (~df["addr1"].isna()).astype(np.int32).values
    has_shipping_dist = (~df["dist1"].isna()).astype(np.int32).values
    shipping_dist = df["dist1"].fillna(-1.0).values.astype(np.float32)

    p_email = df["P_emaildomain"].fillna("").values
    r_email = df["R_emaildomain"].fillna("").values
    email_mismatch = ((p_email != "") & (r_email != "") & (p_email != r_email)).astype(np.int32)
    has_recip_email = (r_email != "").astype(np.int32)

    # 5. Device & Identity Consistency (from train_identity merge)
    has_ident = (~df["DeviceType"].isna() | ~df["DeviceInfo"].isna() | (df.get("id_01") is not None and ~df["id_01"].isna())).astype(np.int32).values
    is_mobile = (df.get("DeviceType", pd.Series(dtype=object)) == "mobile").astype(np.int32).values
    is_desktop = (df.get("DeviceType", pd.Series(dtype=object)) == "desktop").astype(np.int32).values
    has_device_info = (~df.get("DeviceInfo", pd.Series(dtype=object)).isna()).astype(np.int32).values
    browser_known = (~df.get("id_31", pd.Series(dtype=object)).isna()).astype(np.int32).values
    os_known = (~df.get("id_30", pd.Series(dtype=object)).isna()).astype(np.int32).values

    # Proxy flag: id_12 or id_15 or id_16
    id_12 = df.get("id_12", pd.Series(dtype=object)).fillna("").astype(str).str.lower()
    id_15 = df.get("id_15", pd.Series(dtype=object)).fillna("").astype(str).str.lower()
    ip_proxy = ((id_12 == "proxy") | (id_15 == "new")).astype(np.int32).values

    # 6. AVS & Match Verification Signals (M1-M9)
    def parse_m(col: str) -> np.ndarray:
        s = df.get(col, pd.Series(dtype=object))
        arr = np.full(n, -1, dtype=np.int32)
        arr[s == "T"] = 1
        arr[s == "F"] = 0
        return arr

    avs_addr = parse_m("M1")
    avs_bill = parse_m("M2")
    avs_ship = parse_m("M3")

    # Count of 'T' across available M1..M9
    total_m_matches = np.zeros(n, dtype=np.int32)
    for m_col in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]:
        if m_col in df.columns:
            total_m_matches += (df[m_col] == "T").astype(np.int32).values

    # 7. Card & Historical Context (C1, C2, C13, D1, D2)
    c1 = df.get("C1", pd.Series(0, index=df.index)).fillna(0).values.astype(np.float32)
    c2 = df.get("C2", pd.Series(0, index=df.index)).fillna(0).values.astype(np.float32)
    c13 = df.get("C13", pd.Series(0, index=df.index)).fillna(0).values.astype(np.float32)
    d1 = df.get("D1", pd.Series(-1, index=df.index)).fillna(-1).values.astype(np.float32)
    d2 = df.get("D2", pd.Series(-1, index=df.index)).fillna(-1).values.astype(np.float32)

    # 8. Temporal Context
    hour = ((times // SECONDS_IN_HOUR) % 24).astype(np.int32)
    is_night = ((hour >= 0) & (hour <= 5)).astype(np.int32)
    dow = ((times // SECONDS_IN_DAY) % 7).astype(np.int32)

    # Assemble into DataFrame
    feat_dict = {
        "card_txn_count_24h": card_count_24h,
        "card_txn_count_7d": card_count_7d,
        "card_txn_amt_sum_24h": card_amt_sum_24h,
        "card_time_since_last_txn_hours": card_time_since_last,
        "email_domain_txn_count_24h": email_count_24h,
        "card_amt_zscore": card_amt_zscore,
        "amt_to_card_mean_ratio": amt_to_card_mean_ratio,
        "amt_log": amt_log,
        "amt_is_round_dollar": amt_is_round,
        "amt_cents": amt_cents,
        "ProductCD_train_fraud_rate": prod_risk,
        "addr_card_country_mismatch": country_mismatch,
        "has_billing_addr": has_billing_addr,
        "has_shipping_dist": has_shipping_dist,
        "shipping_billing_dist": shipping_dist,
        "email_domain_mismatch": email_mismatch,
        "has_recipient_email": has_recip_email,
        "has_identity_data": has_ident,
        "is_mobile_device": is_mobile,
        "is_desktop_device": is_desktop,
        "has_device_info": has_device_info,
        "browser_is_known": browser_known,
        "os_is_known": os_known,
        "ip_proxy_risk_flag": ip_proxy,
        "avs_address_match": avs_addr,
        "avs_billing_match": avs_bill,
        "avs_shipping_match": avs_ship,
        "total_verification_matches": total_m_matches,
        "card_address_count_C1": c1,
        "transaction_count_C2": c2,
        "email_count_C13": c13,
        "time_delta_card_creation_D1": d1,
        "time_delta_prev_txn_D2": d2,
        "hour_of_day": hour,
        "is_night_transaction": is_night,
        "day_of_week": dow,
    }

    out_df = pd.DataFrame(feat_dict, index=df.index)
    print(f"Engineered {len(FEATURE_NAMES)} features for {n:,} rows in {time.time()-t0:.2f}s")
    return out_df
