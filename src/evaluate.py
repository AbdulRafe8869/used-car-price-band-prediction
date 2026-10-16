"""Evaluate the saved models on the held-out TEST set (used once, here).

    python -m src.evaluate

Metrics for predicting the exact selling price (regression):
  R2                       share of the price variation explained (1 = perfect, 0 = no better than the average)
  MAE / RMSE               typical / root-mean-square error in rupees
  MAPE / median % error    error as a percentage of the real price (fair to cheap and expensive cars alike)
  within +-10 % / +-20 %   share of cars priced to within that percentage of the real price
plus bootstrap confidence intervals, errors by price range, feature importance
and a check of the "likely price range" shown in the app.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as cfg
from . import data_prep as dp
from . import pipeline as pl

MODEL_KEYS = ("dummy", "linear_regression", "random_forest")
REAL_MODELS = ("linear_regression", "random_forest")


# ------------------------------------------------------------------ metrics
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    signed = (y_pred - y_true) / y_true
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_log_price": float(r2_score(np.log(y_true), np.log(np.clip(y_pred, 1, None)))),
        "mae_rs": float(mean_absolute_error(y_true, y_pred)),
        "rmse_rs": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "rmsle": float(np.sqrt(mean_squared_error(np.log(y_true), np.log(np.clip(y_pred, 1, None))))),
        "within_10pct": float((ape <= 0.10).mean()),
        "within_20pct": float((ape <= 0.20).mean()),
        "bias_median_signed_pct": float(np.median(signed)),      # negative = tends to under-price
        "share_overpriced": float((signed > 0).mean()),
    }


def errors_by_price_range(y_true, preds: dict) -> dict:
    """Where are the errors?  Metrics inside each real-price range (analysis only)."""
    y_true = np.asarray(y_true, float)
    rng = dp.price_range_label(y_true)
    out = {}
    for label in cfg.ANALYSIS_LABELS:
        m = rng == label
        out[label] = {"n": int(m.sum())}
        for k, p in preds.items():
            ape = np.abs(p[m] - y_true[m]) / y_true[m]
            out[label][k] = {"mae_rs": float(np.abs(p[m] - y_true[m]).mean()),
                             "median_ape": float(np.median(ape)), "mape": float(ape.mean())}
    return out


def bootstrap_ci(y_true, preds: dict, n_boot: int = 2000, seed: int = cfg.RANDOM_STATE) -> dict:
    """95% bootstrap intervals for R2, MAE and median % error, and for RF minus Linear (paired)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = {k: {"r2": np.empty(n_boot), "mae_rs": np.empty(n_boot), "median_ape": np.empty(n_boot)} for k in preds}
    for b in range(n_boot):
        yt = y_true[idx[b]]
        for k, p in preds.items():
            pb = p[idx[b]]
            stats[k]["r2"][b] = r2_score(yt, pb)
            stats[k]["mae_rs"][b] = np.abs(pb - yt).mean()
            stats[k]["median_ape"][b] = np.median(np.abs(pb - yt) / yt)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {k: {m: ci(v) for m, v in d.items()} for k, d in stats.items()}
    if "random_forest" in stats and "linear_regression" in stats:
        out["rf_minus_linear"] = {}
        for m in ("r2", "mae_rs", "median_ape"):
            diff = stats["random_forest"][m] - stats["linear_regression"][m]
            out["rf_minus_linear"][m] = {"mean": float(diff.mean()), "ci95": ci(diff)}
        better = stats["random_forest"]["mae_rs"] < stats["linear_regression"]["mae_rs"]
        out["rf_minus_linear"]["share_of_resamples_rf_has_lower_mae"] = float(better.mean())
    return out


# ------------------------------------------------------------- importances
def feature_group(name: str, input_features: list[str]) -> str:
    """Map a one-hot / numeric output column back to the engineered feature it came from."""
    for feat in sorted(input_features, key=len, reverse=True):     # "brand_tier" before "brand"
        if name == feat or name.startswith(feat + "_"):
            return feat
    return cfg.TEXT_FEATURE                                        # keyword flags have no prefix


def grouped_impurity_importance(rf_pipeline) -> pd.Series:
    """Random Forest (variance-reduction) importance summed over the one-hot columns of each feature."""
    names = pl.transformed_feature_names(rf_pipeline)
    imp = pl.final_estimator(rf_pipeline).feature_importances_
    groups = [feature_group(n, cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES) for n in names]
    return pd.Series(imp, index=names).groupby(groups).sum().sort_values(ascending=False)


def raw_permutation_importance(pipeline, X, y, n_repeats: int = 20) -> pd.DataFrame:
    """Shuffle each RAW input column on the test set and see how much the error (MAE, Rs) grows."""
    r = permutation_importance(pipeline, X, y, scoring="neg_mean_absolute_error", n_repeats=n_repeats,
                               random_state=cfg.RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "mae_increase_rs": r.importances_mean, "std": r.importances_std})
            .sort_values("mae_increase_rs", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------- main
def load_models() -> dict:
    return {
        "dummy": joblib.load(cfg.DUMMY_MODEL_PATH),
        "linear_regression": joblib.load(cfg.LINEAR_MODEL_PATH),
        "random_forest": joblib.load(cfg.RF_MODEL_PATH),
    }


def evaluate_all(save: bool = True) -> dict:
    train_df, test_df, report = dp.load_and_split()
    X_tr, y_tr = dp.xy(train_df)
    X_te, y_te = dp.xy(test_df)
    models = load_models()
    meta = json.loads(cfg.METADATA_PATH.read_text())

    results, preds = {}, {}
    for name, model in models.items():
        preds[name] = model.predict(X_te)
        results[name] = regression_metrics(y_te, preds[name])
        train_m = regression_metrics(y_tr, model.predict(X_tr))
        results[name]["train_r2"] = train_m["r2"]
        results[name]["train_mape"] = train_m["mape"]
        results[name]["overfit_gap_r2"] = train_m["r2"] - results[name]["r2"]
    for name in REAL_MODELS:
        results[name]["cv_rmsle_best"] = meta["cv_best_rmsle"][name]
        results[name]["cv_r2_best"] = meta["cv_best_r2"][name]
        results[name]["cv_mae_best_rs"] = meta["cv_best_mae"][name]
        results[name]["best_params"] = meta["best_params"][name]
        iv = meta["price_interval"][name]
        lo, hi = preds[name] * iv["lower_multiplier"], preds[name] * iv["upper_multiplier"]
        results[name]["likely_range"] = {
            "target_coverage": iv["coverage_target"],
            "test_coverage": float(((y_te >= lo) & (y_te <= hi)).mean()),
            "lower_multiplier": iv["lower_multiplier"], "upper_multiplier": iv["upper_multiplier"],
        }

    importance = {
        "rf_impurity_grouped": grouped_impurity_importance(models["random_forest"]).round(5).to_dict(),
        "rf_permutation_raw": raw_permutation_importance(models["random_forest"], X_te, y_te).round(1).to_dict("records"),
        "linear_permutation_raw": raw_permutation_importance(models["linear_regression"], X_te, y_te).round(1).to_dict("records"),
    }

    # Model selection uses the CROSS-VALIDATION score (never the test set)
    best_model = min(REAL_MODELS, key=lambda k: results[k]["cv_rmsle_best"])

    out = {
        "best_model": best_model,
        "best_model_selected_by": "lowest 5-fold cross-validated RMSLE on the training set",
        "dataset": {**{k: report[k] for k in ("rows_raw", "duplicates_removed", "rows_clean", "rows_train", "rows_test")},
                    "price_summary": report["price_summary"]},
        "models": results,
        "bootstrap_95ci": bootstrap_ci(y_te, {k: preds[k] for k in REAL_MODELS}),
        "errors_by_price_range": errors_by_price_range(y_te, {k: preds[k] for k in REAL_MODELS}),
        "importance": importance,
    }
    if save:
        cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.METRICS_PATH.write_text(json.dumps(out, indent=2))
        rows = []
        for k in MODEL_KEYS:
            m = results[k]
            rows.append({"model": k, "r2": m["r2"], "mae_rs": m["mae_rs"], "rmse_rs": m["rmse_rs"], "mape": m["mape"],
                         "median_ape": m["median_ape"], "within_10pct": m["within_10pct"], "within_20pct": m["within_20pct"]})
        pd.DataFrame(rows).round(4).to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)
        table = test_df[cfg.RAW_FEATURES].copy()
        table["actual_price"] = y_te.astype(int)
        for k in REAL_MODELS:
            table[f"pred_{k}"] = np.round(preds[k], -2).astype(int)
        pd.DataFrame(table).to_csv(cfg.REPORTS_DIR / "test_predictions.csv", index=False)
    return out


def print_summary(out: dict) -> None:
    print(f"{'model':20s} {'R2':>6s} {'MAE Rs':>9s} {'RMSE Rs':>9s} {'MAPE':>6s} {'medAPE':>7s} {'+-10%':>6s} {'+-20%':>6s} {'trainR2':>8s}")
    for k, m in out["models"].items():
        print(f"{k:20s} {m['r2']:6.3f} {m['mae_rs']:9,.0f} {m['rmse_rs']:9,.0f} {m['mape']:6.3f} {m['median_ape']:7.3f} "
              f"{m['within_10pct']:6.3f} {m['within_20pct']:6.3f} {m['train_r2']:8.3f}")
    for k in REAL_MODELS:
        lr = out["models"][k]["likely_range"]
        print(f"{k}: likely range x{lr['lower_multiplier']:.2f}..x{lr['upper_multiplier']:.2f} covers {lr['test_coverage']:.1%} of test cars (target {lr['target_coverage']:.0%})")
    print("bootstrap 95% CI:", json.dumps({k: v for k, v in out["bootstrap_95ci"].items() if k != "rf_minus_linear"}, default=lambda x: round(x, 3))[:400])


if __name__ == "__main__":
    print_summary(evaluate_all())
