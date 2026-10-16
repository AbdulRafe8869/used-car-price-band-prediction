"""Bonus experiments that go beyond the core project.

    python -m src.experiments

  A. Feature-group ablation          which inputs does the model really need?       (train-set CV)
  B. Richer features (Car details v3)  do engine size / power / mileage / seats help?
  C. External validation (Car details v4)  a different listing source with higher prices
  D. Why the log of the price?       the same models trained on the raw price instead

Core results (metrics.json) never depend on this file.  The test split is only
used for the final numbers in B, C and D; A uses cross-validation on the
training split.
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from . import config as cfg
from . import data_prep as dp
from . import pipeline as pl
from .features import FeatureEngineer

KEYS = ("linear_regression", "random_forest")


def _best_params() -> dict:
    meta = json.loads(cfg.METADATA_PATH.read_text())
    strip = lambda d: {k.replace("model__regressor__", ""): v for k, v in d.items()}
    return {k: strip(v) for k, v in meta["best_params"].items()}


def _estimators(params: dict, log_target: bool = True) -> dict:
    """Ridge and Random Forest regressors with the tuned hyper-parameters from the main project."""
    lin = Ridge(random_state=cfg.RANDOM_STATE, **params["linear_regression"])
    rf = RandomForestRegressor(random_state=cfg.RANDOM_STATE, n_jobs=-1, **params["random_forest"])
    wrap = pl.log_price if log_target else (lambda e: e)
    return {"linear_regression": wrap(lin), "random_forest": wrap(rf)}


def _models(params: dict, numeric=None, categorical=None, use_keywords=True, fe=None, log_target=True) -> dict:
    out = {}
    for key, est in _estimators(params, log_target).items():
        pipe = pl.build_pipeline(est, numeric, categorical, use_keywords)
        if fe is not None:
            pipe.steps[0] = ("features", fe)
        out[key] = pipe
    return out


def _median_ape(y_true, y_pred) -> float:
    return float(np.median(np.abs(y_pred - y_true) / y_true))


MEDIAN_APE_SCORER = make_scorer(_median_ape, greater_is_better=False)


# ---------------------------------------------------------------- A. ablation
ABLATION_GROUPS = {
    "Name keywords (variant words)": dict(use_keywords=False),
    "Brand tier": dict(drop_cat=["brand_tier"]),
    "Fuel, seller type, transmission": dict(drop_cat=["fuel", "seller_type", "transmission"]),
    "Owner count": dict(drop_num=["owner_count"]),
    "Age and usage (age, km, km per year)": dict(drop_num=["car_age", "km_driven", "km_per_year"]),
    "Car identity (brand, model, tier, keywords)": dict(drop_cat=["brand_tier", "brand", "model"], use_keywords=False),
}


def ablation() -> pd.DataFrame:
    """Remove one feature group at a time; delta > 0 means the typical % error got WORSE."""
    train_df, _, _ = dp.load_and_split()
    X, y = dp.xy(train_df)
    params = _best_params()
    cv = KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    def score(kw):
        num = [c for c in cfg.NUMERIC_FEATURES if c not in kw.get("drop_num", [])]
        cat = [c for c in cfg.CATEGORICAL_FEATURES if c not in kw.get("drop_cat", [])]
        res = {}
        for key, pipe in _models(params, num, cat, kw.get("use_keywords", True)).items():
            res[key] = float(-cross_val_score(pipe, X, y, cv=cv, scoring=MEDIAN_APE_SCORER).mean())
        return res

    full = score({})
    rows = [{"group": "Full model (all features)", **{f"cv_medape_{k}": v for k, v in full.items()},
             **{f"delta_{k}": 0.0 for k in full}}]
    for name, kw in ABLATION_GROUPS.items():
        s = score(kw)
        rows.append({"group": f"Without: {name}", **{f"cv_medape_{k}": v for k, v in s.items()},
                     **{f"delta_{k}": s[k] - full[k] for k in s}})
    return pd.DataFrame(rows)


# --------------------------------------------------- B. richer features (v3)
EXTRA_NUMERIC = ["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]


def _first_number(s):
    m = re.search(r"\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else np.nan


def load_v3() -> pd.DataFrame:
    df = pd.read_csv(cfg.V3_CSV)
    df["mileage_kmpl"] = df["mileage"].map(_first_number)
    df["engine_cc"] = df["engine"].map(_first_number)
    df["max_power_bhp"] = df["max_power"].map(_first_number)
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce")
    return df.drop(columns=["mileage", "engine", "max_power", "torque"])


class ExtendedFeatureEngineer(FeatureEngineer):
    """Standard engineered features + four numeric specification columns passed through."""

    def transform(self, X):
        out = super().transform(X)
        X = X.reset_index(drop=True)
        for c in EXTRA_NUMERIC:
            out[c] = pd.to_numeric(X[c], errors="coerce")
        return out


def richer_features(n_boot: int = 2000) -> dict:
    df, rep = dp.clean(load_v3())
    train_df, test_df = dp.split(df)
    cols = cfg.RAW_FEATURES + EXTRA_NUMERIC
    Xtr_b, y_tr = dp.xy(train_df)
    Xte_b, y_te = dp.xy(test_df)
    Xtr_e, Xte_e = train_df[cols], test_df[cols]
    params = _best_params()
    out = {"rows_raw": rep["rows_raw"], "duplicates_removed": rep["duplicates_removed"],
           "rows_clean": rep["rows_clean"], "rows_train": len(train_df), "rows_test": len(test_df),
           "missing_spec_values": int(df[EXTRA_NUMERIC].isna().any(axis=1).sum()), "results": {}}
    base_models = _models(params)
    ext_models = _models(params, numeric=cfg.NUMERIC_FEATURES + EXTRA_NUMERIC, fe=ExtendedFeatureEngineer())
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.integers(0, len(y_te), size=(n_boot, len(y_te)))
    for key in base_models:
        base_models[key].fit(Xtr_b, y_tr)
        ext_models[key].fit(Xtr_e, y_tr)
        pb, pe = base_models[key].predict(Xte_b), ext_models[key].predict(Xte_e)
        ab, ae = np.abs(pb - y_te) / y_te, np.abs(pe - y_te) / y_te
        diff = ab[idx].mean(axis=1) - ae[idx].mean(axis=1)           # > 0: extended features have the lower error
        out["results"][key] = {
            "base_r2": float(r2_score(y_te, pb)), "extended_r2": float(r2_score(y_te, pe)),
            "base_mae_rs": float(mean_absolute_error(y_te, pb)), "extended_mae_rs": float(mean_absolute_error(y_te, pe)),
            "base_mape": float(ab.mean()), "extended_mape": float(ae.mean()),
            "mape_reduction_points": float(ab.mean() - ae.mean()) * 100,
            "mape_reduction_ci95_points": [float(np.percentile(diff, 2.5)) * 100, float(np.percentile(diff, 97.5)) * 100],
        }
    return out


# ------------------------------------------- C. external validation (v4)
V4_FUEL = {"Petrol": "Petrol", "Diesel": "Diesel", "CNG": "CNG", "LPG": "LPG", "Electric": "Electric",
           "Hybrid": "Petrol", "CNG + CNG": "CNG", "Petrol + CNG": "CNG", "Petrol + LPG": "LPG"}
V4_OWNER = {"First": "First Owner", "Second": "Second Owner", "Third": "Third Owner",
            "Fourth": "Fourth & Above Owner", "4 or More": "Fourth & Above Owner"}
V4_SELLER = {"Individual": "Individual", "Corporate": "Dealer", "Commercial Registration": "Dealer"}


def load_v4_as_main_schema() -> pd.DataFrame:
    v = pd.read_csv(cfg.V4_CSV)
    v["Make"] = v["Make"].replace({"Maruti Suzuki": "Maruti"})
    out = pd.DataFrame({
        "name": (v["Make"].astype(str) + " " + v["Model"].astype(str)),
        "year": v["Year"], cfg.TARGET_COL: v["Price"], "km_driven": v["Kilometer"],
        "fuel": v["Fuel Type"].map(V4_FUEL), "seller_type": v["Seller Type"].map(V4_SELLER),
        "transmission": v["Transmission"], "owner": v["Owner"].map(V4_OWNER),
    })
    return out.dropna().drop_duplicates().reset_index(drop=True)


def external_validation() -> dict:
    v4 = load_v4_as_main_schema()
    v4 = v4[(v4["year"] <= 2022) & (v4["km_driven"] < 1_000_000)].copy()
    v4["year"] = v4["year"].clip(upper=cfg.REFERENCE_YEAR)      # model measures age up to the 2020 snapshot
    y = v4[cfg.TARGET_COL].to_numpy(float)
    X = v4[cfg.RAW_FEATURES]
    main_prices = pd.read_csv(cfg.PROCESSED_DIR / "cleaned_data.csv")[cfg.TARGET_COL]
    test_metrics = json.loads(cfg.METRICS_PATH.read_text())["models"]
    out = {"rows": int(len(v4)),
           "median_price_v4": float(np.median(y)), "median_price_main": float(main_prices.median()),
           "price_quantiles_v4": [float(q) for q in np.quantile(y, [0.1, 0.25, 0.5, 0.75, 0.9])],
           "price_quantiles_main": [float(q) for q in np.quantile(main_prices, [0.1, 0.25, 0.5, 0.75, 0.9])],
           "models": {}}
    for key, path in (("linear_regression", cfg.LINEAR_MODEL_PATH), ("random_forest", cfg.RF_MODEL_PATH)):
        pred = joblib.load(path).predict(X)
        ape = np.abs(pred - y) / y
        signed = (pred - y) / y
        out["models"][key] = {
            "r2": float(r2_score(y, pred)), "mae_rs": float(mean_absolute_error(y, pred)),
            "median_ape": float(np.median(ape)), "mape": float(ape.mean()),
            "within_20pct": float((ape <= 0.2).mean()),
            "bias_median_signed_pct": float(np.median(signed)),
            "share_underpriced": float((pred < y).mean()),
            "test_median_ape_main": test_metrics[key]["median_ape"],
            "test_bias_main": test_metrics[key]["bias_median_signed_pct"],
        }
    return out


# ----------------------------------------------- D. why the log of the price
def log_vs_raw_target() -> dict:
    train_df, test_df, _ = dp.load_and_split()
    X_tr, y_tr = dp.xy(train_df)
    X_te, y_te = dp.xy(test_df)
    params = _best_params()
    out = {}
    for label, use_log in (("log_price", True), ("raw_price", False)):
        out[label] = {}
        for key, pipe in _models(params, log_target=use_log).items():
            pred = pipe.fit(X_tr, y_tr).predict(X_te)
            ape = np.abs(pred - y_te) / y_te
            out[label][key] = {"r2": float(r2_score(y_te, pred)), "mae_rs": float(mean_absolute_error(y_te, pred)),
                               "median_ape": float(np.median(ape)), "mape": float(ape.mean()),
                               "negative_predictions": int((pred <= 0).sum())}
    return out


def main() -> dict:
    print("A. ablation ...", flush=True)
    abl = ablation()
    print(abl.round(4).to_string(index=False), flush=True)
    print("B. richer features (v3) ...", flush=True)
    rich = richer_features()
    print(json.dumps(rich["results"], indent=1), flush=True)
    print("C. external validation (v4) ...", flush=True)
    ext = external_validation()
    print(json.dumps(ext, indent=1), flush=True)
    print("D. log price vs raw price ...", flush=True)
    lograw = log_vs_raw_target()
    print(json.dumps(lograw, indent=1), flush=True)
    result = {"ablation": abl.to_dict("records"), "richer_features_v3": rich, "external_validation_v4": ext,
              "log_vs_raw_target": lograw}
    (cfg.REPORTS_DIR / "experiments.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
