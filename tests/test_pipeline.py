"""Fast checks that guard the things a reviewer is most likely to ask about:
no data leakage, a fair split, prices that come out in rupees, and a model that behaves sensibly.

    python -m pytest -q
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data_prep as dp
from src import predict as pr
from src.features import FeatureEngineer, parse_name


# ------------------------------------------------------------ data / split
@pytest.fixture(scope="module")
def split():
    return dp.load_and_split()


def test_cleaning_numbers_match_the_report(split):
    train_df, test_df, rep = split
    assert rep["rows_raw"] == 4340
    assert rep["rows_clean"] == len(train_df) + len(test_df)
    assert rep["rows_raw"] - rep["duplicates_removed"] == rep["rows_clean"]


def test_no_identical_listing_in_both_train_and_test(split):
    train_df, test_df, _ = split
    cols = cfg.RAW_FEATURES + [cfg.TARGET_COL]
    overlap = set(map(tuple, train_df[cols].values)) & set(map(tuple, test_df[cols].values))
    assert not overlap, "duplicates leaked across the split"


def test_split_is_80_20_and_balanced_by_price(split):
    train_df, test_df, _ = split
    assert abs(len(test_df) / (len(train_df) + len(test_df)) - cfg.TEST_SIZE) < 0.005
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    tr, te = train_df[cfg.TARGET_COL].quantile(q).to_numpy(), test_df[cfg.TARGET_COL].quantile(q).to_numpy()
    assert np.all(np.abs(te / tr - 1) < 0.08), "test prices are spread differently from training prices"


def test_target_is_the_exact_price_in_rupees(split):
    train_df, _, _ = split
    X, y = dp.xy(train_df)
    assert list(X.columns) == cfg.RAW_FEATURES
    assert y.dtype == float and y.min() >= 20_000 and "band" not in train_df.columns


def test_price_ranges_are_used_for_analysis_only():
    labels = dp.price_range_label([100_000, 250_000, 499_999, 500_000, 1_000_000, 8_900_000])
    assert list(labels) == ["Below Rs 2.5 L", "Rs 2.5 - 5 L", "Rs 2.5 - 5 L", "Rs 5 - 10 L", "Rs 10 L and above",
                            "Rs 10 L and above"]


# --------------------------------------------------------------- features
@pytest.mark.parametrize("name, expected", [
    ("Maruti Swift Dzire VDI", ("Maruti", "Maruti Swift")),
    ("Land Rover Discovery S 2.0", ("Land Rover", "Land Rover Discovery")),
    ("OpelCorsa 1.4 GL", ("Opel", "Opel Corsa")),
    ("Ford Ecosport 1.5", ("Ford", "Ford Ecosport")),
])
def test_parse_name(name, expected):
    assert parse_name(name) == expected


def test_feature_engineer_handles_new_cars_and_zero_age():
    row = pd.DataFrame([{"name": "Honda City", "year": 2020, "km_driven": 500, "fuel": "Petrol",
                         "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"}])
    out = FeatureEngineer().fit(row).transform(row)
    assert out.loc[0, "car_age"] == 0
    assert np.isfinite(out.loc[0, "km_per_year"])          # no division by zero
    assert out.loc[0, "brand_tier"] == "Mid-range"


# ------------------------------------------------------------ saved models
@pytest.fixture(scope="module")
def models():
    return {"linear": joblib.load(cfg.LINEAR_MODEL_PATH), "rf": joblib.load(cfg.RF_MODEL_PATH)}


def _row(name, year, km, fuel="Petrol", seller="Individual", trans="Manual", owner="First Owner"):
    return pd.DataFrame([{"name": name, "year": year, "km_driven": km, "fuel": fuel,
                          "seller_type": seller, "transmission": trans, "owner": owner}])[cfg.RAW_FEATURES]


def test_predictions_are_plain_rupees(models):
    for m in models.values():
        price = pr.predict_price(m, _row("Maruti Swift VDI", 2014, 70_000))
        assert 100_000 < price < 1_500_000                  # a 2014 Swift is a few lakh, not a log value


def test_unseen_brand_and_model_do_not_crash(models):
    for m in models.values():
        price = pr.predict_price(m, _row("Tesla Roadster X", 2020, 1_000, fuel="Electric"))
        assert np.isfinite(price) and price > 0


def test_newer_low_km_car_costs_more_than_old_high_km_car(models):
    new = _row("Honda City 1.5 V", 2019, 8_000)
    old = _row("Honda City 1.5 V", 2008, 150_000, owner="Third Owner")
    for m in models.values():
        assert pr.predict_price(m, new) > 2 * pr.predict_price(m, old)


def test_explanation_has_one_row_per_input_group(models):
    data = pd.read_csv(cfg.PROCESSED_DIR / "cleaned_data.csv")
    eff = pr.explain_by_occlusion(models["rf"], _row("Hyundai I20 Sportz", 2016, 45_000), pr.typical_reference(data))
    assert set(eff["input"]) == set(pr.INPUT_GROUPS) and np.isfinite(eff["effect_rs"]).all()


def test_indian_rupee_formatting_and_range():
    assert pr.inr(462_300) == "Rs 4,62,000" and pr.inr(12_345_678) == "Rs 1,23,46,000"
    lo, hi = pr.likely_range(500_000, {"lower_multiplier": 0.7, "upper_multiplier": 1.4})
    assert (lo, hi) == (350_000, 700_000)


# ----------------------------------------------------------- saved results
def test_saved_metrics_match_the_dataset_and_beat_the_baseline(split):
    _, test_df, _ = split
    metrics = json.loads(cfg.METRICS_PATH.read_text())
    assert metrics["dataset"]["rows_test"] == len(test_df)
    m = metrics["models"]
    assert m["random_forest"]["n"] == len(test_df)
    for k in ("linear_regression", "random_forest"):
        assert m[k]["r2"] > 0.7 and m[k]["mae_rs"] < 0.5 * m["dummy"]["mae_rs"]


def test_likely_range_covers_roughly_the_promised_share_of_test_cars():
    metrics = json.loads(cfg.METRICS_PATH.read_text())
    for k in ("linear_regression", "random_forest"):
        cov = metrics["models"][k]["likely_range"]["test_coverage"]
        assert abs(cov - cfg.INTERVAL_COVERAGE) < 0.08, (k, cov)
