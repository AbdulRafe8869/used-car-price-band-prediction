"""Loading, cleaning and splitting the CarDekho data.

Order matters here and is deliberate:

    load -> clean (missing, duplicates, invalid) -> split

Duplicates are removed *before* splitting.  If they were removed afterwards,
identical listings would sit in both the training and the test set and the
test scores would be inflated.

The target is the exact selling price in rupees.  (The models work on the
logarithm of the price internally; see pipeline.py.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_raw(path=cfg.MAIN_CSV) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded from Kaggle."""
    return pd.read_csv(path)


def price_range_label(price) -> np.ndarray:
    """Rupee price -> analysis range label (used only to see where the errors are)."""
    code = np.digitize(np.asarray(price, dtype=float), cfg.ANALYSIS_EDGES, right=False)
    return np.asarray(cfg.ANALYSIS_LABELS, dtype=object)[code]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning steps and return (clean_df, report).

    The report records what each step found so the numbers can be shown in the
    notebook, README and presentation.
    """
    report: dict = {"rows_raw": int(len(df))}
    df = df.copy()

    # 1. tidy text (stray spaces would create fake categories)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # 2. missing values
    report["missing_per_column"] = {c: int(v) for c, v in df.isna().sum().items()}
    report["missing_total"] = int(df.isna().sum().sum())
    before = len(df)
    df = df.dropna(subset=[cfg.TARGET_COL])            # a row without a price is useless
    report["rows_dropped_missing_target"] = int(before - len(df))
    # (missing feature values, if any, are imputed inside the Pipeline)

    # 3. duplicate listings
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))
    report["duplicates_pct"] = round(100 * report["duplicates_removed"] / report["rows_raw"], 2)

    # 4. impossible values (none expected, but checked and logged)
    invalid = (
        (df[cfg.TARGET_COL] <= 0)
        | (df["km_driven"] < 0)
        | (df["year"] > cfg.REFERENCE_YEAR)
        | (df["year"] < cfg.MIN_YEAR)
    )
    report["invalid_rows_removed"] = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    price = df[cfg.TARGET_COL].astype(float)
    report["price_summary"] = {
        "min": float(price.min()), "q1": float(price.quantile(0.25)), "median": float(price.median()),
        "mean": float(price.mean()), "q3": float(price.quantile(0.75)), "max": float(price.max()),
        "skewness_raw": float(price.skew()), "skewness_log": float(np.log(price).skew()),
    }

    # plain python objects for the text columns (keeps sklearn / joblib happy)
    for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
        df[col] = df[col].astype(object)
    return df, report


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80:20 split that keeps the price spread the same in both parts.

    The cars are sorted into 10 equal-sized price groups (deciles) and each
    group is split 80:20, so cheap, mid-range and luxury cars appear in the
    test set in the same proportion as in the training set.
    """
    deciles = pd.qcut(df[cfg.TARGET_COL], cfg.SPLIT_PRICE_QUANTILES, labels=False, duplicates="drop")
    train_df, test_df = train_test_split(
        df, test_size=cfg.TEST_SIZE, stratify=deciles, random_state=cfg.RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_and_split(path=cfg.MAIN_CSV):
    """Convenience: raw CSV -> (train_df, test_df, cleaning_report)."""
    clean_df, report = clean(load_raw(path))
    train_df, test_df = split(clean_df)
    report["rows_train"], report["rows_test"] = int(len(train_df)), int(len(test_df))
    return train_df, test_df, report


def xy(df: pd.DataFrame):
    """Raw feature frame and the target price in rupees (float)."""
    return df[cfg.RAW_FEATURES], df[cfg.TARGET_COL].to_numpy(dtype=float)
