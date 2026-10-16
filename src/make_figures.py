"""Regenerate every figure in reports/figures.

    python -m src.make_figures

Needs the trained models and reports/metrics.json (run src.train and
src.evaluate first).  Bonus-experiment figures are drawn only if
reports/experiments.json exists (src.experiments).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve

from . import config as cfg
from . import data_prep as dp
from . import visualize as viz


def compute_learning_curves(models: dict, X, y):
    """Mean absolute % error of the training folds and of the held-out folds, for growing training sizes."""
    cv = KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    out = {}
    for key in ("linear_regression", "random_forest"):
        n, tr, va = learning_curve(models[key], X, y, cv=cv, train_sizes=np.linspace(0.15, 1.0, 6),
                                   scoring="neg_mean_absolute_percentage_error", n_jobs=-1,
                                   random_state=cfg.RANDOM_STATE)[:3]
        out[key] = (n, -tr.mean(axis=1), -va.mean(axis=1), va.std(axis=1))
    return out


def main() -> None:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_df, _ = dp.clean(dp.load_raw())
    train_df, test_df = dp.split(clean_df)
    X_tr, y_tr = dp.xy(train_df)
    X_te, y_te = dp.xy(test_df)
    metrics = json.loads(cfg.METRICS_PATH.read_text())
    models = {"linear_regression": joblib.load(cfg.LINEAR_MODEL_PATH), "random_forest": joblib.load(cfg.RF_MODEL_PATH)}
    preds = {k: m.predict(X_te) for k, m in models.items()}

    figs = {
        "01_price_distribution.png": viz.fig_price_distribution(clean_df),
        "02_target_transform.png": viz.fig_target_transform(clean_df),
        "03_price_drivers.png": viz.fig_price_drivers(clean_df),
        "04_price_by_brand.png": viz.fig_price_by_brand(clean_df),
        "05_outliers.png": viz.fig_outliers(clean_df, train_df),
        "06_correlation.png": viz.fig_correlation(clean_df),
        "07_predicted_vs_actual.png": viz.fig_predicted_vs_actual(y_te, preds, metrics["models"]),
        "08_model_comparison.png": viz.fig_model_comparison(metrics["models"]),
        "09_error_distribution.png": viz.fig_error_distribution(y_te, preds),
        "10_feature_importance.png": viz.fig_feature_importance(metrics["importance"]),
        "12_error_by_price_range.png": viz.fig_error_by_price_range(metrics["errors_by_price_range"]),
    }
    cv_tables = {k: pd.read_csv(cfg.REPORTS_DIR / f"cv_results_{k}.csv") for k in ("linear_regression", "random_forest")}
    figs["13_grid_search.png"] = viz.fig_grid_search(cv_tables)
    print("computing learning curves ...", flush=True)
    figs["11_learning_curves.png"] = viz.fig_learning_curve(compute_learning_curves(models, X_tr, y_tr))

    exp_path = cfg.REPORTS_DIR / "experiments.json"
    if exp_path.exists():
        exp = json.loads(exp_path.read_text())
        figs["14_ablation.png"] = viz.fig_ablation(pd.DataFrame(exp["ablation"]))
        figs["15_richer_features_v3.png"] = viz.fig_richer_features(exp["richer_features_v3"])
        figs["16_external_validation_v4.png"] = viz.fig_external_validation(exp["external_validation_v4"])
        figs["17_log_vs_raw_target.png"] = viz.fig_log_vs_raw(exp["log_vs_raw_target"])

    for name, fig in figs.items():
        viz.save_fig(fig, name)
        print("saved", name)
        viz.plt.close(fig)


if __name__ == "__main__":
    main()
