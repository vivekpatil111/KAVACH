"""Data loader and merger for the IEEE-CIS Fraud Detection dataset.

Handles auto-detection of file formats (.csv, .xlsx, etc.), robust left-joining on
TransactionID, identity column normalization, and diagnostic dataset profiling.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Tuple

import pandas as pd


def _find_file(raw_dir: str, prefix: str) -> str:
    """Auto-detect file with matching prefix and supported extension."""
    candidates = glob.glob(os.path.join(raw_dir, f"{prefix}.*"))
    if not candidates:
        sub_dir = os.path.join(raw_dir, "ieee_cis")
        if os.path.isdir(sub_dir):
            candidates = glob.glob(os.path.join(sub_dir, f"{prefix}.*"))
    for ext in (".csv", ".xlsx", ".xls", ".parquet"):
        for c in candidates:
            if c.lower().endswith(ext):
                return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find file matching '{prefix}.*' in {raw_dir} or {raw_dir}/ieee_cis")


def _load_single_file(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load single tabular file based on detected extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, usecols=usecols)
    elif ext == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    else:
        # Fallback to read_csv
        return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_and_merge_data(
    raw_dir: str = "data/raw",
    load_test: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train and test transaction + identity datasets, left-join on TransactionID.

    Returns:
        train_merged (pd.DataFrame): merged train dataset
        test_merged (pd.DataFrame | None): merged test dataset if load_test is True
    """
    print("=" * 72)
    print("TASK 1: DATA LOADING & MERGING (IEEE-CIS FRAUD DATASET)")
    print("=" * 72)

    # 1. Locate files
    train_tx_path = _find_file(raw_dir, "train_transaction")
    train_id_path = _find_file(raw_dir, "train_identity")
    print(f"Found Train Transaction: {train_tx_path}")
    print(f"Found Train Identity:    {train_id_path}")

    # 2. Load Train
    t0 = time.time()
    print("\nLoading train_transaction...")
    train_tx = _load_single_file(train_tx_path)
    print(f"Loaded train_transaction in {time.time()-t0:.2f}s | Shape: {train_tx.shape}")

    t1 = time.time()
    print("Loading train_identity...")
    train_id = _load_single_file(train_id_path)
    print(f"Loaded train_identity in {time.time()-t1:.2f}s | Shape: {train_id.shape}")

    # 3. Merge Train on TransactionID (left join to preserve all transactions)
    t2 = time.time()
    print("Merging train_transaction + train_identity on TransactionID (left join)...")
    train_merged = train_tx.merge(train_id, on="TransactionID", how="left")
    print(f"Train merged in {time.time()-t2:.2f}s | Result Shape: {train_merged.shape}")

    # 4. Profile Train
    mem_train_mb = train_merged.memory_usage(deep=True).sum() / (1024 ** 2)
    fraud_counts = train_merged["isFraud"].value_counts().to_dict()
    fraud_pct = train_merged["isFraud"].mean() * 100

    print("\n--- TRAIN DATASET SUMMARY ---")
    print(f"Total Rows:           {len(train_merged):,}")
    print(f"Total Columns:        {train_merged.shape[1]}")
    print(f"Memory Usage:         {mem_train_mb:.2f} MB ({mem_train_mb / 1024:.2f} GB)")
    print(f"Legitimate (isFraud=0): {fraud_counts.get(0, 0):,} ({100 - fraud_pct:.3f}%)")
    print(f"Fraudulent (isFraud=1): {fraud_counts.get(1, 0):,} ({fraud_pct:.3f}%)")
    print(f"Overall Fraud Rate:   {fraud_pct:.4f}% (Base rate: 1 in {100/fraud_pct:.1f} transactions)")

    # 5. Missing Value Statistics
    missing_series = train_merged.isna().mean() * 100
    high_missing = missing_series[missing_series > 50]
    zero_missing = missing_series[missing_series == 0]
    print(f"\nMissing Value Analysis:")
    print(f"  Columns with 0% missing:     {len(zero_missing)} / {train_merged.shape[1]}")
    print(f"  Columns with >50% missing:   {len(high_missing)} / {train_merged.shape[1]}")
    print(f"  Mean column missingness:     {missing_series.mean():.2f}%")

    # Sample top missing
    print("\nSample Columns with Highest Missingness:")
    for col, pct in missing_series.sort_values(ascending=False).head(10).items():
        print(f"  {col:20s}: {pct:.2f}% missing")

    # 6. Load Test Dataset (if requested, for inference demonstration)
    test_merged = None
    if load_test:
        test_tx_path = _find_file(raw_dir, "test_transaction")
        test_id_path = _find_file(raw_dir, "test_identity")
        print(f"\nFound Test Transaction:  {test_tx_path}")
        print(f"Found Test Identity:     {test_id_path}")

        t3 = time.time()
        print("\nLoading test_transaction...")
        test_tx = _load_single_file(test_tx_path)
        print(f"Loaded test_transaction in {time.time()-t3:.2f}s | Shape: {test_tx.shape}")

        t4 = time.time()
        print("Loading test_identity...")
        test_id = _load_single_file(test_id_path)
        # Normalize test identity column names: replace '-' with '_' (e.g. id-01 -> id_01)
        test_id.columns = [c.replace("-", "_") for c in test_id.columns]
        print(f"Loaded test_identity in {time.time()-t4:.2f}s | Shape: {test_id.shape}")

        t5 = time.time()
        print("Merging test_transaction + test_identity on TransactionID (left join)...")
        test_merged = test_tx.merge(test_id, on="TransactionID", how="left")
        print(f"Test merged in {time.time()-t5:.2f}s | Result Shape: {test_merged.shape}")

        mem_test_mb = test_merged.memory_usage(deep=True).sum() / (1024 ** 2)
        print("\n--- TEST DATASET SUMMARY ---")
        print(f"Total Rows:           {len(test_merged):,}")
        print(f"Total Columns:        {test_merged.shape[1]}")
        print(f"Memory Usage:         {mem_test_mb:.2f} MB ({mem_test_mb / 1024:.2f} GB)")
        print("Note: Test set is unlabelled (isFraud not present) — reserved for demo inference only.")

    return train_merged, test_merged


if __name__ == "__main__":
    train_df, test_df = load_and_merge_data()
