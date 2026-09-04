"""End-to-end execution pipeline for Chargeback Defense & Return-Risk Scorer.

Executes:
1. Data loading & left-join merge
2. Temporal split (70% train, 15% val, 15% held-out test)
3. Causal feature engineering without lookahead leakage
4. Model training (XGBoost) with early stopping on validation split
5. Rigorous evaluation strictly on held-out temporal test split
6. Single-feature dominance check (must not exceed 50% of total importance)
7. Artifact persistence (model, feature list, eval report, data summary)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from chargeback_defense.data_loader import load_and_merge_data
from chargeback_defense.feature_engineering import (
    FEATURE_NAMES,
    compute_product_risk,
    engineer_features,
)


def run_pipeline(raw_dir: str = "data/raw", models_dir: str = "models", reports_dir: str = "reports") -> dict:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    total_start = time.time()

    # =========================================================================
    # STEP 1: DATA LOADING & MERGE
    # =========================================================================
    train_merged, test_merged = load_and_merge_data(raw_dir=raw_dir, load_test=True)

    # =========================================================================
    # STEP 2: TEMPORAL SPLIT (NOT RANDOM)
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 2: CHRONOLOGICAL TEMPORAL SPLIT")
    print("=" * 72)

    # Sort strictly by TransactionDT
    print("Sorting dataset chronologically by TransactionDT...")
    t_sort = time.time()
    train_merged = train_merged.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    print(f"Sorted in {time.time()-t_sort:.2f}s")

    n_total = len(train_merged)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    split_indices = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_total),
    }

    train_slice = train_merged.iloc[:n_train]
    val_slice = train_merged.iloc[n_train:n_train + n_val]
    test_slice = train_merged.iloc[n_train + n_val:]

    def get_split_stats(name: str, df_split: pd.DataFrame) -> dict:
        cnt = len(df_split)
        fraud_cnt = int(df_split["isFraud"].sum())
        fraud_rate = float(df_split["isFraud"].mean())
        dt_min = int(df_split["TransactionDT"].min())
        dt_max = int(df_split["TransactionDT"].max())
        day_min = round(dt_min / 86400.0, 2)
        day_max = round(dt_max / 86400.0, 2)
        return {
            "name": name,
            "rows": cnt,
            "fraction": cnt / n_total,
            "fraud_count": fraud_cnt,
            "fraud_rate": fraud_rate,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "day_min": day_min,
            "day_max": day_max,
        }

    stats_train = get_split_stats("Train (70%)", train_slice)
    stats_val = get_split_stats("Validation (15%)", val_slice)
    stats_test = get_split_stats("Held-Out Test (15%)", test_slice)

    print(f"\n{'Split Name':<22} | {'Rows':<9} | {'Share':<6} | {'Fraud Count':<11} | {'Fraud Rate':<10} | {'DT Range (Days)':<20}")
    print("-" * 88)
    for s in [stats_train, stats_val, stats_test]:
        print(f"{s['name']:<22} | {s['rows']:<9,d} | {s['fraction']:<6.1%} | {s['fraud_count']:<11,d} | {s['fraud_rate']:<10.4%} | Day {s['day_min']:<5.1f} - {s['day_max']:<5.1f}")

    # Check for class imbalance shift
    max_rate = max(stats_train["fraud_rate"], stats_val["fraud_rate"], stats_test["fraud_rate"])
    min_rate = min(stats_train["fraud_rate"], stats_val["fraud_rate"], stats_test["fraud_rate"])
    shift_ratio = max_rate / min_rate if min_rate > 0 else 1.0
    print(f"\nClass balance stability check across temporal splits:")
    print(f"  Train: {stats_train['fraud_rate']:.4%} | Val: {stats_val['fraud_rate']:.4%} | Test: {stats_test['fraud_rate']:.4%}")
    print(f"  Shift Ratio (Max/Min): {shift_ratio:.3f}x")
    if shift_ratio < 1.25:
        print("  Status: STABLE — no severe macroeconomic fraud drift observed across splits.")
    else:
        print("  Status: WARNING — notable drift in base rate observed across splits.")

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING (FIRST PRINCIPLES)
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 3: CAUSAL FEATURE ENGINEERING (CHARGEBACK / RETURN-RISK)")
    print("=" * 72)

    # 1. Product Category Risk Encoding: Strictly from train split
    print("Computing ProductCD risk encoding strictly from Train split (zero leakage)...")
    product_risk_map = compute_product_risk(train_slice)
    print("ProductCD smoothed fraud rates (Train only):")
    for prod, r in sorted(product_risk_map.items()):
        print(f"  ProductCD '{prod}': {r:.4%}")

    # 2. Engineer features over full sorted dataset causally
    print("\nExecuting sequential causal feature engineering...")
    features_df = engineer_features(train_merged, product_risk_map)

    # Extract X and y for each split
    X_train = features_df.iloc[:n_train][FEATURE_NAMES]
    y_train = train_slice["isFraud"].values

    X_val = features_df.iloc[n_train:n_train + n_val][FEATURE_NAMES]
    y_val = val_slice["isFraud"].values

    X_test = features_df.iloc[n_train + n_val:][FEATURE_NAMES]
    y_test = test_slice["isFraud"].values

    print(f"Feature matrix shapes:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    # =========================================================================
    # STEP 4: BASELINE MODEL TRAINING (XGBOOST)
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 4: BASELINE MODEL TRAINING & EARLY STOPPING")
    print("=" * 72)

    # Calculate scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos = round(float(neg_count) / max(pos_count, 1), 2)
    print(f"Class imbalance: {neg_count:,} negative / {pos_count:,} positive (scale_pos_weight={scale_pos})")

    # Using balanced hyperparameters:
    # moderate scale_pos_weight (square root of ratio or dampening) to calibrate probabilities
    damped_scale = round(np.sqrt(scale_pos), 2)
    print(f"Using damped scale_pos_weight={damped_scale} for well-calibrated probability scores.")

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        scale_pos_weight=damped_scale,
        eval_metric=["aucpr", "auc"],
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )

    t_train = time.time()
    print("Fitting XGBClassifier with validation monitoring...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"Training completed in {time.time()-t_train:.2f}s | Best Iteration: {best_iter}")

    # =========================================================================
    # STEP 5: EVALUATION ON HELD-OUT TEMPORAL TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 5: RIGOROUS EVALUATION ON HELD-OUT TEMPORAL TEST SET")
    print("=" * 72)

    test_scores = model.predict_proba(X_test)[:, 1]

    # PR-AUC and ROC-AUC
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, test_scores)
    pr_auc_score = float(auc(pr_rec, pr_prec))
    roc_auc_val = float(roc_auc_score(y_test, test_scores))
    test_base_rate = float(y_test.mean())

    print(f"Held-Out Test Set Metrics (N = {len(y_test):,} transactions, base rate = {test_base_rate:.4%}):")
    print(f"  PR-AUC (Precision-Recall Area): {pr_auc_score:.4f} (vs baseline {test_base_rate:.4f})")
    print(f"  ROC-AUC:                        {roc_auc_val:.4f} (vs baseline 0.5000)")

    # Candidate Thresholds Sweep
    candidate_thresholds = [0.50, 0.70, 0.85, 0.90]
    threshold_results = []

    print("\nCandidate Operating Threshold Performance:")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Flagged':<9} | {'TP':<7} | {'FP':<7} | {'FN':<7}")
    print("-" * 88)

    for thr in candidate_thresholds:
        preds = (test_scores >= thr).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())

        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        flagged = int(preds.sum())

        threshold_results.append({
            "threshold": thr,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "flagged": flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
        print(f"{thr:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {flagged:<9,d} | {tp:<7,d} | {fp:<7,d} | {fn:<7,d}")

    # Choose Recommended Operating Threshold
    # For Chargeback Evidence Responder:
    # A threshold of 0.70 provides an optimal trade-off: high precision to avoid generating
    # unnecessary evidence defense packets, while maintaining high recall on actual chargebacks.
    recommended_thr = 0.70
    rec_res = next(r for r in threshold_results if r["threshold"] == recommended_thr)
    conf_matrix = {
        "true_negative": rec_res["tn"],
        "false_positive": rec_res["fp"],
        "false_negative": rec_res["fn"],
        "true_positive": rec_res["tp"],
    }

    print(f"\nRecommended Operating Threshold: {recommended_thr:.2f}")
    print(f"  Confusion Matrix: TP={rec_res['tp']:,}, FP={rec_res['fp']:,}, FN={rec_res['fn']:,}, TN={rec_res['tn']:,}")
    print(f"  Precision: {rec_res['precision']:.2%} | Recall: {rec_res['recall']:.2%} | F1: {rec_res['f1']:.4f}")

    # =========================================================================
    # STEP 6: FEATURE IMPORTANCE DISTRIBUTION & DOMINANCE CHECK
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 6: FEATURE IMPORTANCE DISTRIBUTION & SINGLE-FEATURE CHECK")
    print("=" * 72)

    importances = model.feature_importances_
    total_gain = float(importances.sum())
    norm_importances = importances / total_gain if total_gain > 0 else importances

    feat_ranking = sorted(
        zip(FEATURE_NAMES, (float(v) for v in norm_importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )

    print(f"{'Rank':<5} | {'Feature Name':<32} | {'Importance':<12} | {'Bar'}")
    print("-" * 72)
    for idx, (fname, imp) in enumerate(feat_ranking, 1):
        bar = "#" * int(imp * 60)
        print(f"{idx:<5} | {fname:<32} | {imp:<12.4%} | {bar}")

    top_feature_name, top_feature_share = feat_ranking[0]
    single_dominance_flag = bool(top_feature_share > 0.50)

    print("\n--- SINGLE-FEATURE DOMINANCE CHECK ---")
    print(f"Top Feature:        {top_feature_name} ({top_feature_share:.2%})")
    print(f"Dominance Ceiling:  50.00%")
    if single_dominance_flag:
        print(f"CHECK STATUS:       FAILED (Feature '{top_feature_name}' exceeds 50% threshold — fragile reliance!)")
    else:
        print(f"CHECK STATUS:       PASSED (No feature exceeds 50% — genuine multi-signal robustness achieved!)")

    # =========================================================================
    # STEP 7: SAVE ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 72)
    print("TASK 7: SAVING ARTIFACTS & EVALUATION REPORT")
    print("=" * 72)

    model_path = os.path.join(models_dir, "chargeback_xgb.json")
    model.save_model(model_path)
    print(f"Saved trained model to: {model_path}")

    feat_path = os.path.join(models_dir, "feature_list.json")
    with open(feat_path, "w") as fh:
        json.dump(FEATURE_NAMES, fh, indent=2)
    print(f"Saved feature list to:  {feat_path}")

    eval_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "IEEE-CIS Fraud Detection (Real anonymized Vesta/IEEE-CIS transactions)",
        "split_type": "Temporal (70% train, 15% validation, 15% test by TransactionDT)",
        "splits": {
            "train": stats_train,
            "validation": stats_val,
            "test": stats_test,
        },
        "held_out_metrics": {
            "pr_auc": round(pr_auc_score, 6),
            "roc_auc": round(roc_auc_val, 6),
            "test_base_rate": round(test_base_rate, 6),
            "recommended_threshold": recommended_thr,
            "confusion_matrix": conf_matrix,
            "threshold_sweep": threshold_results,
        },
        "feature_dominance_check": {
            "passed": not single_dominance_flag,
            "top_feature": top_feature_name,
            "top_feature_importance": round(top_feature_share, 4),
            "threshold": 0.50,
        },
        "feature_importances": [
            {"feature": f, "importance": round(v, 6)} for f, v in feat_ranking
        ],
    }

    report_path = os.path.join(reports_dir, "eval_report.json")
    with open(report_path, "w") as fh:
        json.dump(eval_report, fh, indent=2)
    print(f"Saved eval report to:   {report_path}")

    print(f"\nPipeline finished in {time.time()-total_start:.2f} seconds.")
    return eval_report


if __name__ == "__main__":
    run_pipeline()
