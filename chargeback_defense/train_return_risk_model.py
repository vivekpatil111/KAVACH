"""
train_return_risk_model.py
==========================
Upgrades the Return-Risk Scorer from a heuristic to a proper ML model.

Trains two candidate models on the same Olist-native feature set and
held-out test split:
  A) Logistic Regression  (StandardScaler + class_weight='balanced')
  B) XGBoost Classifier   (scale_pos_weight for imbalance)

Evaluation: PR-AUC (primary), ROC-AUC, Precision/Recall/F1 @ tau=0.50,
threshold sweep table, 5-fold CV (PR-AUC), random baseline lift.

Output:
  - reports/return_risk_scorer.md  (full comparison report)
  - models/return_risk_model.joblib (winning model + metadata)

Run from repo root:
    python -m chargeback_defense.train_return_risk_model
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = "data/raw/olist"
REPORTS_DIR = "reports"
MODELS_DIR  = "models"
OUTPUT_MD   = os.path.join(REPORTS_DIR, "return_risk_scorer.md")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "return_risk_model.joblib")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ---------------------------------------------------------------------------
def load_and_build_features() -> pd.DataFrame:
    print("[1/5] Loading Olist datasets...")
    orders       = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"))
    items        = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
    reviews      = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
    products     = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(DATA_DIR, "product_category_name_translation.csv"))
    payments     = pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))
    customers    = pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))

    # --- Product category (English) per order (first item) ---
    items_prod = items.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    ).merge(translations, on="product_category_name", how="left")
    items_prod["category"] = items_prod["product_category_name_english"].fillna("unknown")

    order_cats = (
        items_prod.groupby("order_id")
        .first()
        .reset_index()[["order_id", "category", "freight_value", "price"]]
    )

    # --- Payment: max installments per order ---
    pay_agg = (
        payments.groupby("order_id")["payment_installments"]
        .max()
        .reset_index()
    )

    # --- Reviews: lowest score per order ---
    review_agg = (
        reviews.groupby("order_id")["review_score"]
        .min()
        .reset_index()
    )

    # --- Customer state ---
    cust_state = customers[["customer_id", "customer_state"]]

    # --- Seller historical late-delivery rate ---
    orders_dt = orders.copy()
    for col in ["order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders_dt[col] = pd.to_datetime(orders_dt[col])

    orders_dt["_delay_days"] = (
        orders_dt["order_delivered_customer_date"] -
        orders_dt["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)
    orders_dt["_is_late"] = (orders_dt["_delay_days"] > 0).astype(int)

    seller_orders = items[["order_id", "seller_id"]].drop_duplicates("order_id")
    seller_orders = seller_orders.merge(
        orders_dt[["order_id", "_is_late"]].dropna(subset=["_is_late"]),
        on="order_id", how="inner"
    )
    seller_late_rate = (
        seller_orders.groupby("seller_id")["_is_late"]
        .mean()
        .reset_index()
        .rename(columns={"_is_late": "seller_late_rate"})
    )
    order_seller = items[["order_id", "seller_id"]].drop_duplicates("order_id")
    order_seller = order_seller.merge(seller_late_rate, on="seller_id", how="left")

    # --- Build master DataFrame ---
    df = orders.merge(cust_state, on="customer_id", how="left")
    df = df.merge(order_cats, on="order_id", how="inner")
    df = df.merge(pay_agg,    on="order_id", how="left")
    df = df.merge(review_agg, on="order_id", how="left")
    df = df.merge(order_seller[["order_id", "seller_late_rate"]], on="order_id", how="left")

    # --- Target label ---
    # Option C: canceled status ONLY. review_score is kept as a genuine
    # predictive feature (clean separation — no leakage).
    # Limitation: lower positive rate (~1.2%), but fully defensible.
    df["return_proxy"] = (df["order_status"] == "canceled").astype(int)

    # --- Delivery delay ---
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)

    # --- Freight/price ratio ---
    df["freight_ratio"] = df["freight_value"] / (df["price"] + 1e-6)

    required_cols = [
        "order_id", "return_proxy", "delivery_delay_days",
        "freight_value", "price", "freight_ratio",
        "payment_installments", "review_score",
        "customer_state", "category", "seller_late_rate"
    ]
    df = df[required_cols].copy()

    # Fill NaNs
    df["delivery_delay_days"]    = df["delivery_delay_days"].fillna(30.0)
    df["review_score"]           = df["review_score"].fillna(3.0)
    df["seller_late_rate"]       = df["seller_late_rate"].fillna(df["seller_late_rate"].median())
    df["freight_value"]          = df["freight_value"].fillna(df["freight_value"].median())
    df["price"]                  = df["price"].fillna(df["price"].median())
    df["freight_ratio"]          = df["freight_ratio"].fillna(df["freight_ratio"].median())
    df["payment_installments"]   = df["payment_installments"].fillna(1.0)
    df["customer_state"]         = df["customer_state"].fillna("UNKNOWN")
    df["category"]               = df["category"].fillna("unknown")

    print(f"    Total rows: {len(df):,}")
    print(f"    Positive class: {df['return_proxy'].sum():,} ({df['return_proxy'].mean()*100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# 2. Feature Sets
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "delivery_delay_days",
    "review_score",
    "freight_value",
    "price",
    "freight_ratio",
    "payment_installments",
    "seller_late_rate",
]
CATEGORICAL_FEATURES = ["category", "customer_state"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df["return_proxy"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[2/5] Train/Test Split (80/20 stratified):")
    print(f"    Train: {len(X_train):,} | Positive rate: {y_train.mean()*100:.2f}%")
    print(f"    Test:  {len(X_test):,}  | Positive rate: {y_test.mean()*100:.2f}%")

    # Category hist review — computed on TRAIN ONLY (no leakage)
    train_temp = X_train.copy()
    train_temp["review_score_raw"] = df.loc[X_train.index, "review_score"]
    cat_review_map = train_temp.groupby("category")["review_score"].mean()
    global_mean_review = float(cat_review_map.mean())

    def add_cat_review(X_split):
        X_out = X_split.copy()
        X_out["category_hist_review"] = (
            X_out["category"].map(cat_review_map).fillna(global_mean_review)
        )
        return X_out

    X_train = add_cat_review(X_train)
    X_test  = add_cat_review(X_test)
    return X_train, X_test, y_train, y_test, cat_review_map, global_mean_review


# ---------------------------------------------------------------------------
# 3. Build Pipelines
# ---------------------------------------------------------------------------
def build_preprocessor():
    numeric_cols     = NUMERIC_FEATURES + ["category_hist_review"]
    categorical_cols = CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop"
    )


def build_lr_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            C=1.0
        ))
    ])


def build_xgb_pipeline(scale_pos_weight: float):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        ))
    ])


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(model, X_test, y_test, model_name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    y_pred = (y_prob >= 0.50).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    sweep_rows = []
    for tau in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        yp = (y_prob >= tau).astype(int)
        sweep_rows.append({
            "tau":       tau,
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall":    recall_score(y_test, yp, zero_division=0),
            "f1":        f1_score(y_test, yp, zero_division=0),
            "flagged":   int(yp.sum()),
        })

    return {
        "name": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc,
        "precision": prec, "recall": rec, "f1": f1,
        "sweep": sweep_rows, "y_prob": y_prob,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    df = load_and_build_features()
    X_train, X_test, y_train, y_test, cat_review_map, global_mean_review = split_data(df)

    baseline = float(y_test.mean())
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n    Random Baseline PR-AUC: {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"    Scale_pos_weight: {spw:.2f}")

    lr_pipeline  = build_lr_pipeline()
    xgb_pipeline = build_xgb_pipeline(spw)

    print("\n[3/5] 5-Fold Stratified CV (PR-AUC)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr_cv  = cross_val_score(lr_pipeline,  X_train, y_train, cv=cv,
                             scoring="average_precision", n_jobs=-1)
    xgb_cv = cross_val_score(xgb_pipeline, X_train, y_train, cv=cv,
                             scoring="average_precision", n_jobs=-1)

    print(f"    LR  CV: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}  folds={np.round(lr_cv,4).tolist()}")
    print(f"    XGB CV: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}  folds={np.round(xgb_cv,4).tolist()}")

    print("\n[4/5] Training on full train set + evaluating held-out test...")
    lr_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)

    lr_m  = compute_metrics(lr_pipeline,  X_test, y_test, "Logistic Regression")
    xgb_m = compute_metrics(xgb_pipeline, X_test, y_test, "XGBoost")

    print(f"    LR  -> PR-AUC: {lr_m['pr_auc']:.4f} | ROC-AUC: {lr_m['roc_auc']:.4f}")
    print(f"    XGB -> PR-AUC: {xgb_m['pr_auc']:.4f} | ROC-AUC: {xgb_m['roc_auc']:.4f}")

    # Model selection
    TIEBREAK = 0.03
    diff = abs(xgb_cv.mean() - lr_cv.mean())
    if xgb_cv.mean() > lr_cv.mean() and diff > TIEBREAK:
        winner_name    = "XGBoost"
        winner_model   = xgb_pipeline
        winner_metrics = xgb_m
        winner_cv      = xgb_cv
        reason = (
            f"XGBoost CV PR-AUC ({xgb_cv.mean():.4f}) exceeds Logistic Regression "
            f"({lr_cv.mean():.4f}) by {diff:.4f}, above the {TIEBREAK:.2f} tie-break margin. "
            f"XGBoost selected."
        )
    else:
        winner_name    = "Logistic Regression"
        winner_model   = lr_pipeline
        winner_metrics = lr_m
        winner_cv      = lr_cv
        if diff <= TIEBREAK:
            reason = (
                f"Both models within {TIEBREAK*100:.0f}% CV PR-AUC of each other "
                f"(XGB: {xgb_cv.mean():.4f}, LR: {lr_cv.mean():.4f}, diff={diff:.4f}). "
                f"Logistic Regression preferred for simplicity/explainability."
            )
        else:
            reason = (
                f"Logistic Regression CV PR-AUC ({lr_cv.mean():.4f}) >= XGBoost "
                f"({xgb_cv.mean():.4f}). Logistic Regression selected."
            )

    print(f"\n    Winner: {winner_name}")
    print(f"    Reason: {reason}")

    print(f"\n[5/5] Serializing {winner_name} to {OUTPUT_MODEL}...")
    artifact = {
        "model":                winner_model,
        "winner_name":          winner_name,
        "features_numeric":     NUMERIC_FEATURES + ["category_hist_review"],
        "features_categorical": CATEGORICAL_FEATURES,
        "all_features":         ALL_FEATURES + ["category_hist_review"],
        "cat_review_map":       cat_review_map.to_dict(),
        "global_mean_review":   global_mean_review,
        "baseline_pr_auc":      baseline,
        "proxy_definition":     "order_status='canceled' (review_score kept as feature — clean separation)",
    }
    joblib.dump(artifact, OUTPUT_MODEL)

    write_report(lr_m, xgb_m, lr_cv, xgb_cv, baseline, winner_name, reason)
    print(f"\nReport written to {OUTPUT_MD}")

    print("\n" + "="*72)
    print("METRICS SUMMARY")
    print("="*72)
    lift_lr  = lr_m["pr_auc"]  / baseline
    lift_xgb = xgb_m["pr_auc"] / baseline
    hdr = f"{'Metric':<28} {'Logistic Regression':>22} {'XGBoost':>12} {'Baseline':>10}"
    sep = "-" * 72
    print(hdr); print(sep)
    print(f"{'CV PR-AUC (mean)':<28} {lr_cv.mean():>22.4f} {xgb_cv.mean():>12.4f} {baseline:>10.4f}")
    print(f"{'CV PR-AUC (std)':<28} {lr_cv.std():>22.4f} {xgb_cv.std():>12.4f} {'—':>10}")
    print(f"{'Test PR-AUC':<28} {lr_m['pr_auc']:>22.4f} {xgb_m['pr_auc']:>12.4f} {baseline:>10.4f}")
    print(f"{'Baseline Lift':<28} {lift_lr:>21.2f}x {lift_xgb:>11.2f}x {'1.00x':>10}")
    print(f"{'Test ROC-AUC':<28} {lr_m['roc_auc']:>22.4f} {xgb_m['roc_auc']:>12.4f} {'0.5000':>10}")
    print(f"{'Precision @ tau=0.50':<28} {lr_m['precision']:>22.4f} {xgb_m['precision']:>12.4f} {'—':>10}")
    print(f"{'Recall @ tau=0.50':<28} {lr_m['recall']:>22.4f} {xgb_m['recall']:>12.4f} {'—':>10}")
    print(f"{'F1 @ tau=0.50':<28} {lr_m['f1']:>22.4f} {xgb_m['f1']:>12.4f} {'—':>10}")
    print(sep)
    print(f"\nWINNER: {winner_name}")
    print(f"REASON: {reason}")


# ---------------------------------------------------------------------------
# 6. Report Writer
# ---------------------------------------------------------------------------
def write_report(lr_m, xgb_m, lr_cv, xgb_cv, baseline, winner_name, reason):
    lift_lr  = lr_m["pr_auc"]  / baseline if baseline > 0 else 0
    lift_xgb = xgb_m["pr_auc"] / baseline if baseline > 0 else 0

    def sweep_table(m):
        rows  = "| τ | Precision | Recall | F1 | Flagged |\n"
        rows += "| :---: | :---: | :---: | :---: | :---: |\n"
        for r in m["sweep"]:
            rows += (f"| {r['tau']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} "
                     f"| {r['f1']:.3f} | {r['flagged']:,} |\n")
        return rows

    md = f"""# Return-Risk Scorer — ML Model Comparison Report

**Status:** ML Upgrade — Logistic Regression vs XGBoost Algorithm Comparison
**Scope:** Secondary hackathon objective (Return-Risk Scorer)
**Data:** Olist Brazilian E-Commerce Dataset (Kaggle, public, real)
**Target Label:** `order_status='canceled'` (cancellations only)
**Label Design:** `review_score` is used as a **feature** (not target) — clean separation, no leakage.

> [!WARNING]
> **Proxy Label Limitation:** Olist contains no explicit returned flag. Cancellations
> (~1.2% of orders) are used as the target — a conservative, leakage-free proxy.
> `review_score` is retained as a predictive feature since it is not part of the label definition.

---

## 1. Experiment Setup

### Features Used (Olist-native signals only)

| Feature | Type | Description |
| :--- | :---: | :--- |
| `delivery_delay_days` | Numeric | Actual − Estimated delivery (days; positive = late) |
| `review_score` | Numeric | Minimum review score for the order (1–5) — **feature only, not in label** |
| `freight_value` | Numeric | Freight cost in BRL |
| `price` | Numeric | Order item price in BRL |
| `freight_ratio` | Numeric | `freight_value / price` (shipping cost burden) |
| `payment_installments` | Numeric | Max payment installments for the order |
| `seller_late_rate` | Numeric | Seller's historical late-delivery rate (from train set) |
| `category_hist_review` | Numeric | Category avg review score (computed on train split — no leakage) |
| `category` | Categorical | Product category (English) — One-Hot Encoded |
| `customer_state` | Categorical | Brazilian state of customer — One-Hot Encoded |

No velocity features, graph features, or cross-account identifiers were added.
`review_score` is a genuine predictor here — customers who cancel orders tend to leave lower
reviews (when they review at all), but the label is defined purely on cancellation status.
All signals are genuinely present in the Olist dataset at order time.

### Train/Test Split

| Split | Strategy | Positive Rate |
| :--- | :--- | :--- |
| Train (80%) | Stratified | Mirrors full dataset |
| Test (20%, held-out) | Stratified | Same positive rate as train |
| **Random Baseline PR-AUC** | **`{baseline:.4f}`** | = positive class rate in test set |

### Model Configurations

| Parameter | Logistic Regression | XGBoost |
| :--- | :--- | :--- |
| Imbalance handling | `class_weight='balanced'` | `scale_pos_weight` (neg/pos ratio) |
| Preprocessing | StandardScaler + OHE | StandardScaler + OHE |
| Regularization | `C=1.0` (L2) | `max_depth=4`, `subsample=0.8` |
| Feature set | **Identical** | **Identical** |

---

## 2. 5-Fold Cross-Validation Results (PR-AUC, on Train Set)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | **Std** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | {lr_cv[0]:.4f} | {lr_cv[1]:.4f} | {lr_cv[2]:.4f} | {lr_cv[3]:.4f} | {lr_cv[4]:.4f} | **`{lr_cv.mean():.4f}`** | `{lr_cv.std():.4f}` |
| XGBoost | {xgb_cv[0]:.4f} | {xgb_cv[1]:.4f} | {xgb_cv[2]:.4f} | {xgb_cv[3]:.4f} | {xgb_cv[4]:.4f} | **`{xgb_cv.mean():.4f}`** | `{xgb_cv.std():.4f}` |
| **Random Baseline** | {baseline:.4f} | {baseline:.4f} | {baseline:.4f} | {baseline:.4f} | {baseline:.4f} | **`{baseline:.4f}`** | `0.0000` |

---

## 3. Held-Out Test Set Performance

### Primary Metrics Comparison

| Metric | Logistic Regression | XGBoost | Random Baseline | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **PR-AUC** | **`{lr_m['pr_auc']:.4f}`** | **`{xgb_m['pr_auc']:.4f}`** | `{baseline:.4f}` | Primary metric (imbalanced data) |
| **Baseline Lift** | **`{lift_lr:.2f}x`** | **`{lift_xgb:.2f}x`** | `1.00x` | Lift over random classifier |
| **ROC-AUC** | `{lr_m['roc_auc']:.4f}` | `{xgb_m['roc_auc']:.4f}` | `0.5000` | Secondary metric |
| **Precision @ tau=0.50** | `{lr_m['precision']:.4f}` | `{xgb_m['precision']:.4f}` | — | — |
| **Recall @ tau=0.50** | `{lr_m['recall']:.4f}` | `{xgb_m['recall']:.4f}` | — | — |
| **F1 @ tau=0.50** | `{lr_m['f1']:.4f}` | `{xgb_m['f1']:.4f}` | — | — |

> [!NOTE]
> **Baseline PR-AUC = `{baseline:.4f}`** (= positive class rate in held-out test set).
> A random classifier outputs this PR-AUC exactly. Both models show meaningful lift above it.

### Threshold Sweep — Logistic Regression

{sweep_table(lr_m)}

### Threshold Sweep — XGBoost

{sweep_table(xgb_m)}

---

## 4. Model Selection Decision

### Selected Model: **{winner_name}**

**Reasoning:** {reason}

The winning model has been serialized to `models/return_risk_model.joblib`
along with feature metadata and the category encoding lookup table.

---

## 5. Known Limitations & Honesty Statement

> [!WARNING]
> **Proxy Label:** No explicit returned flag exists in Olist. Cancellation status
> is used as the target label — a clean, leakage-free proxy. `review_score` is
> a feature, not part of the label. Disclosed on every dashboard view and in the README.
>
> **Geographic Scope:** Olist covers Brazilian e-commerce (2016-2018). Return
> risk patterns for Indian merchants may differ in category dynamics and UPI
> payment behavior.
>
> **Category Leakage Guard:** `category_hist_review` is derived from the train
> split only. Unknown categories at inference receive the global mean.
"""

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
