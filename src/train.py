"""Train and tune the models.

    python -m src.train            # full Grid Search (several minutes)
    python -m src.train --fast     # tiny grids, for a quick smoke test

Steps
  1. load -> clean -> price-balanced 80:20 split                 (data_prep.py)
  2. Baseline, Linear Regression (Ridge), Random Forest          (pipeline.py)
  3. Grid Search with 5-fold CV on the TRAIN set only
  4. "Likely price range": how far real prices sit from the prediction, measured on
     out-of-fold predictions of the training set (never on the test set)
  5. save fitted pipelines, CV tables, split files and app metadata
The test set is not touched here; it is used once, in evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from . import config as cfg
from . import data_prep as dp
from . import pipeline as pl
from .features import parse_name

SCORING = {
    "rmsle": "neg_root_mean_squared_log_error",      # error on the log scale ~ typical % error
    "mae": "neg_mean_absolute_error",                 # average error in rupees
    "r2": "r2",
}
REFIT = "rmsle"


def make_cv() -> KFold:
    return KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)


def grid_search(name: str, pipe, grid: dict, X, y, n_jobs: int = -1) -> GridSearchCV:
    search = GridSearchCV(
        pipe, grid, cv=make_cv(), scoring=SCORING, refit=REFIT,
        n_jobs=n_jobs, return_train_score=True, verbose=0,
    )
    n_fits = int(np.prod([len(v) for v in grid.values()])) * cfg.CV_FOLDS
    print(f"[{name}] Grid Search: {n_fits} fits ({cfg.CV_FOLDS}-fold CV) ...", flush=True)
    t0 = time.time()
    search.fit(X, y)
    print(f"[{name}] done in {time.time() - t0:.0f}s | best CV RMSLE = {-search.best_score_:.4f}", flush=True)
    print(f"[{name}] best params: {search.best_params_}", flush=True)
    return search


def save_cv_table(search: GridSearchCV, path) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "params": res["params"].astype(str),
        "cv_rmsle": -res["mean_test_rmsle"], "cv_rmsle_std": res["std_test_rmsle"],
        "cv_mae_rs": -res["mean_test_mae"], "cv_r2": res["mean_test_r2"],
        "train_r2": res["mean_train_r2"], "rank": res["rank_test_rmsle"],
    }).sort_values("rank")
    table.to_csv(path, index=False)
    return table


def price_interval(estimator, X, y) -> dict:
    """Multipliers giving a 'likely price range' around a prediction.

    ratio = real price / predicted price, measured on out-of-fold predictions
    (every training car is predicted by a model that never saw it).  Taking the
    10th and 90th percentiles of the ratio gives a range that should contain
    about 80 % of cars.  evaluate.py then checks this claim on the test set.
    """
    oof = cross_val_predict(estimator, X, y, cv=make_cv(), n_jobs=1)
    ratio = y / oof
    lo_q = (1 - cfg.INTERVAL_COVERAGE) / 2
    return {"lower_multiplier": float(np.quantile(ratio, lo_q)),
            "upper_multiplier": float(np.quantile(ratio, 1 - lo_q)),
            "coverage_target": cfg.INTERVAL_COVERAGE}


def build_app_metadata(clean_df: pd.DataFrame, report: dict, searches: dict, intervals: dict) -> dict:
    """Everything the Streamlit app needs to build its form, kept next to the model."""
    parsed = clean_df["name"].map(parse_name)
    brand = parsed.map(lambda t: t[0])
    model = parsed.map(lambda t: t[1])
    brand_counts = brand.value_counts()
    model_counts = model.value_counts()
    common_brands = sorted(brand_counts[brand_counts >= cfg.APP_MIN_BRAND_LISTINGS].index)
    brand_of_model = pd.DataFrame({"brand": brand, "model": model}).drop_duplicates("model").set_index("model")["brand"]
    models_by_brand = {b: [] for b in common_brands}
    for m, count in model_counts.items():
        b = brand_of_model[m]
        if count >= cfg.APP_MIN_MODEL_LISTINGS and b in models_by_brand:
            models_by_brand[b].append(m[len(b) + 1:])      # "Maruti Swift" -> "Swift"
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "random_state": cfg.RANDOM_STATE,
        "reference_year": cfg.REFERENCE_YEAR,
        "year_range": [int(clean_df["year"].min()), int(clean_df["year"].max())],
        "km_range": [int(clean_df["km_driven"].min()), int(clean_df["km_driven"].max())],
        "price_range": [float(clean_df[cfg.TARGET_COL].min()), float(clean_df[cfg.TARGET_COL].max())],
        "choices": {
            "fuel": sorted(clean_df["fuel"].unique()),
            "seller_type": sorted(clean_df["seller_type"].unique()),
            "transmission": sorted(clean_df["transmission"].unique()),
            "owner": list(cfg.OWNER_MAP.keys()),
        },
        "brands": common_brands,
        "models_by_brand": {b: sorted(v) for b, v in models_by_brand.items()},
        "cleaning_report": report,
        "best_params": {k: {p: (v if not isinstance(v, np.generic) else v.item()) for p, v in s.best_params_.items()}
                        for k, s in searches.items()},
        "cv_best_rmsle": {k: round(float(-s.best_score_), 4) for k, s in searches.items()},
        "cv_best_r2": {k: round(float(s.cv_results_["mean_test_r2"][s.best_index_]), 4) for k, s in searches.items()},
        "cv_best_mae": {k: round(float(-s.cv_results_["mean_test_mae"][s.best_index_]), 0) for k, s in searches.items()},
        "price_interval": intervals,
    }


def main(fast: bool = False) -> None:
    for d in (cfg.PROCESSED_DIR, cfg.MODELS_DIR, cfg.REPORTS_DIR, cfg.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data -------------------------------------------------------------
    clean_df, report = dp.clean(dp.load_raw())
    train_df, test_df = dp.split(clean_df)
    report["rows_train"], report["rows_test"] = len(train_df), len(test_df)
    clean_df.to_csv(cfg.PROCESSED_DIR / "cleaned_data.csv", index=False)
    train_df.to_csv(cfg.PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(cfg.PROCESSED_DIR / "test.csv", index=False)
    print(f"cleaned {report['rows_raw']} -> {report['rows_clean']} rows "
          f"({report['duplicates_removed']} duplicates removed); train {len(train_df)} / test {len(test_df)}")
    X_train, y_train = dp.xy(train_df)

    # 2-3. models ---------------------------------------------------------
    lin_grid = cfg.LINEAR_GRID_FAST if fast else cfg.LINEAR_GRID
    rf_grid = cfg.RF_GRID_FAST if fast else cfg.RF_GRID
    searches = {
        "linear_regression": grid_search("Linear", pl.make_linear_regression(), lin_grid, X_train, y_train),
        "random_forest": grid_search("RandForest", pl.make_random_forest(), rf_grid, X_train, y_train),
    }
    dummy = pl.make_dummy().fit(X_train, y_train)

    # 4. likely price range -------------------------------------------------
    print("calibrating the likely price range (out-of-fold predictions) ...", flush=True)
    intervals = {k: price_interval(s.best_estimator_, X_train, y_train) for k, s in searches.items()}
    print({k: {m: round(v, 3) for m, v in d.items()} for k, d in intervals.items()})

    # 5. save -------------------------------------------------------------
    joblib.dump(searches["linear_regression"].best_estimator_, cfg.LINEAR_MODEL_PATH, compress=3)
    joblib.dump(searches["random_forest"].best_estimator_, cfg.RF_MODEL_PATH, compress=3)
    joblib.dump(dummy, cfg.DUMMY_MODEL_PATH, compress=3)
    save_cv_table(searches["linear_regression"], cfg.REPORTS_DIR / "cv_results_linear_regression.csv")
    save_cv_table(searches["random_forest"], cfg.REPORTS_DIR / "cv_results_random_forest.csv")
    meta = build_app_metadata(clean_df, report, searches, intervals)
    meta["grid_mode"] = "fast" if fast else "full"
    cfg.METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved models -> {cfg.MODELS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="tiny grids (smoke test only)")
    main(fast=ap.parse_args().fast)
