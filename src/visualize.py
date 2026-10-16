"""All project figures in one place (used by the notebook, evaluate script and app).

Design rules applied (validated with the dataviz palette checker):
  * Model identity  : blue = Random Forest, orange = Linear Regression, gray = baseline
  * Magnitude       : one-hue blue;  polarity: blue <-> red with a gray midpoint
  * Thin bars, hairline solid gridlines, no chart borders, text in ink colours (never series colours)
Every function returns a matplotlib Figure; `save_fig` writes it to reports/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg") if matplotlib.get_backend().lower() in ("", "agg") else None
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator

from . import config as cfg

# ---------------------------------------------------------------- palette
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"      # categorical slots 1, 2 + context gray
RF_COLOR, LIN_COLOR, BASE_COLOR = BLUE, ORANGE, GRAY
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])
MODEL_NAMES = {"random_forest": "Random Forest", "linear_regression": "Linear Regression",
               "dummy": "Average-price baseline"}
MODEL_COLORS = {"random_forest": RF_COLOR, "linear_regression": LIN_COLOR, "dummy": BASE_COLOR}
LAKH_TICKS = [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0, "text.color": INK,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 200,
    })


apply_style()


def _clean(ax, grid: str | None = None):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if grid:
        ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _lakh_axis(ax, axis="x", ticks=LAKH_TICKS):
    """Log axis labelled in Rs lakh (1 lakh = Rs 100,000)."""
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    fmt = FuncFormatter(lambda v, p: f"{v / 1e5:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_formatter(FuncFormatter(lambda *_: ""))


def _pct(v, _p=None) -> str:
    return f"{v:.0%}"


def save_fig(fig, name: str, folder=None):
    folder = folder or cfg.FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return path


def _title(fig, text, sub=None):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    if sub:
        fig.text(0.01, 0.985, sub, ha="left", va="top", fontsize=9.5, color=INK2)


# ------------------------------------------------------------ 1. dataset
def fig_price_distribution(df: pd.DataFrame):
    """Selling price on a log axis with the median and mean marked."""
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    price = df[cfg.TARGET_COL].to_numpy(float)
    bins = np.logspace(np.log10(price.min()), np.log10(price.max()), 46)
    counts, edges = np.histogram(price, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    _lakh_axis(ax, "x", [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000])
    ax.set_xlabel("Selling price in Rs lakh (log scale; 1 lakh = Rs 100,000)")
    ax.set_ylabel("Number of cars")
    _clean(ax, "y")
    top = counts.max()
    med, mean = np.median(price), price.mean()
    ax.axvline(med, ymax=0.86, color=INK, lw=1.2, zorder=4)
    ax.axvline(mean, ymax=0.86, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(med / 1.04, top * 1.12, f"median\nRs {med / 1e5:.1f} L", ha="right", va="top", fontsize=9, color=INK)
    ax.text(mean * 1.04, top * 1.12, f"mean\nRs {mean / 1e5:.1f} L", ha="left", va="top", fontsize=9, color=INK2)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(f"Used-car prices are right-skewed: from Rs {price.min() / 1e5:.2f} L to Rs {price.max() / 1e5:.0f} L ({len(price):,} cars)")
    return fig


def fig_target_transform(df: pd.DataFrame):
    """Raw price vs log(price): why the models learn the logarithm of the price."""
    price = df[cfg.TARGET_COL].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    ax.hist(price / 1e5, bins=60, color=GRAY, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_xlabel("Selling price (Rs lakh)"); ax.set_ylabel("Number of cars")
    ax.set_title(f"Raw price: long right tail (skewness {pd.Series(price).skew():.1f})"); _clean(ax, "y")
    ax = axes[1]
    ax.hist(np.log(price), bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ticks = np.log(LAKH_TICKS)
    ax.set_xticks(ticks, [f"{t / 1e5:g}" for t in LAKH_TICKS])
    ax.set_xlabel("Selling price (Rs lakh, log scale)")
    ax.set_title(f"log(price): almost symmetric (skewness {abs(pd.Series(np.log(price)).skew()):.1f})"); _clean(ax, "y")
    fig.tight_layout(w_pad=3)
    return fig


def fig_price_drivers(df: pd.DataFrame):
    d = df.copy()
    d["car_age"] = cfg.REFERENCE_YEAR - d["year"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    price_l = lambda s: s / 1e5

    ax = axes[0, 0]                                      # age: line
    g = d.groupby("car_age")[cfg.TARGET_COL].agg(["median", "size"])
    g = g[g["size"] >= 15]
    ax.plot(g.index, price_l(g["median"]), color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter(g.index, price_l(g["median"]), s=28, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.set_xlabel("Car age in 2020 (years)"); ax.set_ylabel("Median price (Rs lakh)")
    ax.set_title("Older cars sell for less"); _clean(ax, "y")

    def bars(ax, col, title, order=None, rename=None):
        g = d.groupby(col)[cfg.TARGET_COL].agg(["median", "size"])
        g = g[g["size"] >= 15]
        if order:
            g = g.reindex([o for o in order if o in g.index])
        labels = [rename.get(i, i) if rename else i for i in g.index]
        y = np.arange(len(g))[::-1]
        ax.barh(y, price_l(g["median"]), height=0.5, color=BLUE, zorder=3)
        for yi, (m, n) in zip(y, zip(g["median"], g["size"])):
            ax.text(price_l(m) + 0.15, yi, f"{price_l(m):.1f} L  (n={n:,})", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, labels); ax.set_xlim(0, price_l(g["median"]).max() * 1.5)
        ax.set_xlabel("Median price (Rs lakh)"); ax.set_title(title); _clean(ax, "x")

    bars(axes[0, 1], "fuel", "Diesel cars cost more than petrol", order=["Diesel", "Petrol", "CNG", "LPG"])
    bars(axes[1, 0], "transmission", "Automatics cost far more than manuals")
    bars(axes[1, 1], "owner", "Fewer previous owners, higher price",
         order=["Test Drive Car", "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    _title(fig, "What moves the price of a used car?", None)
    return fig


def fig_price_by_brand(df: pd.DataFrame, top: int = 12):
    """Median price per brand (bar) with the middle 50 % of its prices (line)."""
    from .features import parse_name
    d = df.copy()
    d["brand"] = d["name"].map(lambda n: parse_name(n)[0])
    keep = d["brand"].value_counts().head(top).index
    g = d[d["brand"].isin(keep)].groupby("brand")[cfg.TARGET_COL].agg(
        median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="size").sort_values("median")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = np.arange(len(g))
    ax.barh(y, g["median"] / 1e5, height=0.55, color=BLUE, zorder=3)
    ax.hlines(y, g["q1"] / 1e5, g["q3"] / 1e5, color=INK, lw=1.6, zorder=4)
    for yi, (m, q3) in enumerate(zip(g["median"], g["q3"])):
        ax.text(q3 / 1e5 + 0.4, yi, f"{m / 1e5:.1f} L", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"{b}  (n={int(n):,})" for b, n in zip(g.index, g["n"])])
    ax.set_xlabel("Median selling price, Rs lakh  (black line = middle 50 % of the brand's cars)")
    ax.set_xlim(0, (g["q3"].max() / 1e5) * 1.16)
    _clean(ax, "x")
    ax.set_title("Brand largely decides the price level")
    return fig


def fig_outliers(df: pd.DataFrame, train_df: pd.DataFrame | None = None):
    """Outlier handling: price outliers are real (kept); extreme mileage is clipped at IQR fences."""
    train_df = df if train_df is None else train_df
    km = train_df["km_driven"].to_numpy(float)
    q1, q3 = np.percentile(km, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - cfg.IQR_K * iqr, q3 + cfg.IQR_K * iqr
    raw = df["km_driven"].to_numpy(float)
    clipped = np.clip(raw, lo, hi)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, data, title, color in [
        (axes[0], df[cfg.TARGET_COL].to_numpy(float) / 1e5, "Price (Rs lakh, log): real luxury cars,\nkept; the log scale tames them", GRAY),
        (axes[1], raw / 1e3, "Mileage before clipping\n(thousand km)", GRAY),
        (axes[2], clipped / 1e3, f"Mileage after clipping at\nQ3 + {cfg.IQR_K:g} x IQR = {hi / 1e3:,.0f}k km", BLUE),
    ]:
        bp = ax.boxplot(data, vert=True, widths=0.45, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3.2, markerfacecolor=color, markeredgecolor="none", alpha=0.55),
                        boxprops=dict(facecolor=SURFACE, edgecolor=color, linewidth=1.6),
                        medianprops=dict(color=color, linewidth=2), whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2))
        ax.set_xticks([]); ax.set_title(title, fontsize=10); _clean(ax, "y"); ax.spines["bottom"].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}"))
    axes[1].set_ylim(0, raw.max() / 1e3 * 1.05)
    axes[2].set_ylim(0, raw.max() / 1e3 * 1.05)       # same scale as the middle panel for a fair comparison
    n_clip = int(((raw < lo) | (raw > hi)).sum())
    axes[2].text(0.5, 0.97, f"{n_clip} cars clipped", transform=axes[2].transAxes, ha="center", va="top", color=INK2, fontsize=9)
    fig.tight_layout(w_pad=3)
    return fig


def fig_correlation(df: pd.DataFrame):
    d = pd.DataFrame({
        "Selling price": df[cfg.TARGET_COL],
        "Car age": cfg.REFERENCE_YEAR - df["year"],
        "Km driven": df["km_driven"],
        "Km per year": df["km_driven"] / (cfg.REFERENCE_YEAR - df["year"]).clip(lower=1),
        "Owner count": df["owner"].map(cfg.OWNER_MAP),
    })
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right"); ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=INK if abs(v) < 0.6 else "#ffffff")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.outline.set_visible(False)
    cb.set_label("Spearman correlation (blue = negative, red = positive)", color=INK2, fontsize=8.5)
    ax.set_title("Price falls with age and mileage (rank correlation)")
    return fig


# -------------------------------------------------------- 3. model results
def fig_grid_search(cv_tables: dict):
    """Every Grid Search configuration, ranked by cross-validated RMSLE (lower is better)."""
    fig, axes = plt.subplots(1, len(cv_tables), figsize=(5.2 * len(cv_tables), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, table) in zip(axes, cv_tables.items()):
        t = table.sort_values("cv_rmsle", ascending=True).reset_index(drop=True)
        color = MODEL_COLORS[key]
        ax.errorbar(np.arange(len(t)), t["cv_rmsle"], yerr=t["cv_rmsle_std"], fmt="none", ecolor=GRID, elinewidth=1, zorder=2)
        ax.scatter(np.arange(len(t)), t["cv_rmsle"], s=22, color=MUTED, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter([0], [t.loc[0, "cv_rmsle"]], s=70, color=color, edgecolor=SURFACE, linewidth=1.8, zorder=4)
        ax.annotate(f"best  {t.loc[0, 'cv_rmsle']:.3f}", (0, t.loc[0, "cv_rmsle"]),
                    xytext=(10, -14), textcoords="offset points", fontsize=9.5, color=INK)
        ax.set_title(f"{MODEL_NAMES[key]}: {len(t)} configurations")
        ax.set_xlabel("Configuration rank (best to worst)"); _clean(ax, "y")
    axes[0].set_ylabel("5-fold CV error on log price, RMSLE\n(lower is better; bars = std)")
    fig.tight_layout()
    return fig


def fig_model_comparison(metrics: dict):
    models = ["dummy", "linear_regression", "random_forest"]
    panels = [("r2", "R\u00b2\n(higher is better)", "{:.2f}", None),
              ("median_ape", "Median % error\n(lower is better)", "{:.0%}", _pct),
              ("within_20pct", "Cars priced within \u00b120 %\n(higher is better)", "{:.0%}", _pct),
              ("mae_rs", "Mean error, Rs thousand\n(lower is better)", "{:,.0f}", None)]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
    for ax, (key, title, fmt, axfmt) in zip(axes, panels):
        vals = [metrics[m][key] / (1000 if key == "mae_rs" else 1) for m in models]
        vals = [0.0 if abs(v) < 0.005 else v for v in vals]              # print "0.00", not "-0.00"
        x = np.arange(len(models))
        ax.bar(x, vals, width=0.55, color=[MODEL_COLORS[m] for m in models], zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, max(v, 0) + (max(vals) - min(0, min(vals))) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(x, ["Baseline", "Linear\nRegression", "Random\nForest"], fontsize=8.5)
        if axfmt:
            ax.yaxis.set_major_formatter(FuncFormatter(axfmt))
        ax.set_title(title, fontsize=10); _clean(ax, "y")
        ax.set_ylim(min(0, min(vals)) * 1.3 if min(vals) < 0 else 0, max(vals) * 1.18)
    fig.tight_layout(w_pad=2)
    return fig


def fig_predicted_vs_actual(y_true, preds: dict, metrics: dict):
    """Each dot is one test car: real price (x) against predicted price (y)."""
    y_true = np.asarray(y_true, float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    lim = [y_true.min() * 0.8, y_true.max() * 1.25]
    xs = np.array(lim)
    for ax, key in zip(axes, ("linear_regression", "random_forest")):
        ax.scatter(y_true, preds[key], s=14, color=MODEL_COLORS[key], alpha=0.55, linewidth=0, zorder=3)
        ax.plot(xs, xs, color=INK, lw=1.2, zorder=4)
        ax.plot(xs, xs * 1.2, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.plot(xs, xs * 0.8, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=2)
        _lakh_axis(ax, "x", [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000])
        _lakh_axis(ax, "y", [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000])
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
        m = metrics[key]
        ax.set_title(f"{MODEL_NAMES[key]}: R\u00b2 {m['r2']:.2f}, {m['within_20pct']:.0%} within \u00b120 %")
        ax.set_xlabel("Real selling price (Rs lakh, log scale)"); _clean(ax, "both")
    axes[0].set_ylabel("Predicted price (Rs lakh, log scale)")
    axes[0].text(0.03, 0.97, "black line = perfect prediction\ndashed lines = \u00b120 %", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(w_pad=2)
    return fig


def fig_error_distribution(y_true, preds: dict):
    """How far off is each prediction, as a percentage of the real price?"""
    y_true = np.asarray(y_true, float)
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    lo, hi = -0.6, 1.0
    bins = np.linspace(lo, hi, 41)
    ax.axvspan(-0.2, 0.2, color=GRID, alpha=0.55, zorder=1, linewidth=0)
    for key in ("linear_regression", "random_forest"):
        err = np.clip((preds[key] - y_true) / y_true, lo, hi - 1e-9)
        c, e = np.histogram(err, bins=bins)
        ax.stairs(c, e, color=MODEL_COLORS[key], lw=2, label=MODEL_NAMES[key], zorder=3)
    ax.axvline(0, color=INK, lw=1, zorder=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:+.0%}"))
    ax.set_xlabel("Prediction error as a share of the real price  (negative = too cheap, positive = too expensive)")
    ax.set_ylabel("Number of test cars"); _clean(ax, "y")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    ax.text(0, top * 1.10, "within \u00b120 %", ha="center", va="top", fontsize=9, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=6)
    ax.text(hi, -top * 0.17, "errors beyond the range are counted in the end bars", ha="right", va="top", fontsize=8, color=MUTED)
    ax.legend(loc="upper right")
    med = float(np.median(np.abs(preds["linear_regression"] - y_true) / y_true))
    ax.set_title(f"Half of the predictions are within about {med:.0%} of the real price; the long tail is cars priced too high")
    return fig


def fig_error_by_price_range(by_range: dict):
    """Median % error inside each real-price range (analysis only)."""
    labels = list(by_range)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(labels)); w = 0.34
    for k, key in enumerate(("linear_regression", "random_forest")):
        vals = [by_range[l][key]["median_ape"] for l in labels]
        pos = x + (k - 0.5) * (w + 0.04)
        ax.bar(pos, vals, width=w, color=MODEL_COLORS[key], label=MODEL_NAMES[key], zorder=3)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.005, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{l}\n(n={by_range[l]['n']})" for l in labels])
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_ylabel("Median error as a share of the real price"); ax.set_xlabel("Real selling price of the test car")
    _clean(ax, "y"); ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Typical % error in each price range", pad=28)
    return fig


def fig_feature_importance(importance: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    nice = {"car_age": "Car age", "km_driven": "Km driven", "km_per_year": "Km per year", "owner_count": "Owner count",
            "fuel": "Fuel", "seller_type": "Seller type", "transmission": "Transmission", "brand_tier": "Brand tier",
            "brand": "Brand", "model": "Model", "name_keywords": "Name keywords (variant)"}
    imp = pd.Series(importance["rf_impurity_grouped"]).sort_values()
    ax = axes[0]
    y = np.arange(len(imp))
    ax.barh(y, imp.values, height=0.55, color=BLUE, zorder=3)
    for yi, v in zip(y, imp.values):
        ax.text(v + imp.max() * 0.015, yi, f"{v:.1%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [nice.get(i, i) for i in imp.index]); ax.set_xlim(0, imp.max() * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0%}"))
    ax.set_xlabel("Share of Random Forest importance"); ax.set_title("Inside the forest (variance reduction)"); _clean(ax, "x")

    ax = axes[1]
    rf = pd.DataFrame(importance["rf_permutation_raw"]).set_index("feature")["mae_increase_rs"]
    lin = pd.DataFrame(importance["linear_permutation_raw"]).set_index("feature")["mae_increase_rs"]
    order = rf.sort_values().index
    rawnice = {"name": "Name (brand, model, variant)", "year": "Year", "km_driven": "Km driven", "fuel": "Fuel",
               "seller_type": "Seller type", "transmission": "Transmission", "owner": "Owner"}
    y = np.arange(len(order)); h = 0.3
    ax.barh(y + h / 2 + 0.02, rf[order] / 1000, height=h, color=RF_COLOR, label=MODEL_NAMES["random_forest"], zorder=3)
    ax.barh(y - h / 2 - 0.02, lin[order] / 1000, height=h, color=LIN_COLOR, label=MODEL_NAMES["linear_regression"], zorder=3)
    ax.set_yticks(y, [rawnice.get(i, i) for i in order])
    ax.set_xlabel("Increase in average error when the column is shuffled (Rs thousand)")
    ax.set_title("Permutation importance on the test set"); _clean(ax, "x"); ax.legend(loc="lower right")
    fig.tight_layout(w_pad=3)
    return fig


def fig_learning_curve(curves: dict):
    """curves[key] = (train_sizes, train_mape, cv_mape, cv_std)   (mean absolute % error, lower is better)"""
    fig, axes = plt.subplots(1, len(curves), figsize=(5.6 * len(curves), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, (n, tr, va, vs)) in zip(axes, curves.items()):
        ax.plot(n, tr, color=BLUE, lw=2, solid_capstyle="round", label="Training error", zorder=3)
        ax.plot(n, va, color=ORANGE, lw=2, solid_capstyle="round", label="Cross-validated error", zorder=3)
        ax.fill_between(n, va - vs, va + vs, color=ORANGE, alpha=0.10, linewidth=0)
        for series, color in ((tr, BLUE), (va, ORANGE)):
            ax.scatter([n[-1]], [series[-1]], s=40, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=4)
        ax.set_title(MODEL_NAMES[key]); ax.set_xlabel("Training examples"); _clean(ax, "y"); ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Mean absolute % error (lower is better)")
    fig.tight_layout()
    return fig


# ------------------------------------------------------- bonus experiments
def fig_ablation(table: pd.DataFrame):
    """table columns: group, delta_<model> = change in CV median % error when the feature group is removed."""
    t = table[table["group"] != "Full model (all features)"].copy()
    t["group"] = t["group"].str.replace("Without: ", "", regex=False)
    t = t.sort_values("delta_linear_regression")
    full = table[table["group"] == "Full model (all features)"].iloc[0]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(t)); h = 0.3
    hi = t[["delta_linear_regression", "delta_random_forest"]].max().max() * 100
    for key, off in (("random_forest", +1), ("linear_regression", -1)):
        vals = t[f"delta_{key}"].to_numpy() * 100
        ax.barh(y + off * (h / 2 + 0.02), vals, height=h, color=MODEL_COLORS[key], label=MODEL_NAMES[key], zorder=3)
        for yi, v in zip(y + off * (h / 2 + 0.02), vals):
            ax.text(v + hi * 0.015 if v >= 0 else v - hi * 0.015, yi, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, t["group"]); ax.set_xlim(min(0, t[["delta_linear_regression", "delta_random_forest"]].min().min() * 100 * 1.4), hi * 1.18)
    ax.set_xlabel("Change in 5-fold CV median % error when the feature group is removed (percentage points; right = worse)")
    _clean(ax, "x"); ax.legend(loc="lower right")
    ax.set_title(f"Which features does the model need?  (full-model median error: "
                 f"Linear {full['cv_medape_linear_regression']:.1%}, RF {full['cv_medape_random_forest']:.1%})")
    return fig


def fig_richer_features(res: dict):
    """Base vs extended features on the larger v3 file, for both models (test mean % error)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    keys = ["linear_regression", "random_forest"]
    x = np.arange(2); w = 0.3
    base = [res["results"][k]["base_mape"] for k in keys]
    ext = [res["results"][k]["extended_mape"] for k in keys]
    ax.bar(x - (w / 2 + 0.02), base, width=w, color=GRAY, label="Core 7 features", zorder=3)
    ax.bar(x + (w / 2 + 0.02), ext, width=w, color=BLUE, label="+ engine, power, mileage, seats", zorder=3)
    for xs, vals in ((x - (w / 2 + 0.02), base), (x + (w / 2 + 0.02), ext)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.004, f"{v:.1%}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x, [MODEL_NAMES[k] for k in keys]); ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylabel("Mean absolute % error on the test set (lower is better)"); _clean(ax, "y")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.set_ylim(0, max(base + ext) * 1.2)
    gains = [res["results"][k]["mape_reduction_ci95_points"] for k in keys]
    clear = all(lo > 0 for lo, _ in gains)
    ax.set_title(("Extra specifications lower the error (Car details v3, larger file)" if clear else
                  "Extra specifications change the error only slightly (Car details v3)"), pad=28)
    return fig


def fig_external_validation(ext: dict):
    """Left: price spread of the core data vs the external file. Right: typical % error and bias."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rows = [("Core data (2020)", ext["price_quantiles_main"]), ("External file v4", ext["price_quantiles_v4"])]
    y = np.arange(2)[::-1]
    for yi, (name, q) in zip(y, rows):
        q10, q25, q50, q75, q90 = q
        ax.hlines(yi, q10, q90, color=AXIS, lw=2, zorder=2)
        ax.hlines(yi, q25, q75, color=BLUE, lw=9, zorder=3)
        ax.scatter([q50], [yi], s=60, color=INK, zorder=4)
        ax.text(q50, yi + 0.22, f"median Rs {q50 / 1e5:.1f} L", ha="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_ylim(-0.6, 1.6)
    _lakh_axis(ax, "x", [100_000, 250_000, 500_000, 1_000_000, 2_500_000])
    ax.set_xlabel("Selling price in Rs lakh (log scale; thick = middle 50 %, thin = 10th to 90th percentile)"); _clean(ax, "x")
    ax.set_title(f"v4 listings are pricier (median Rs {ext['median_price_v4'] / 1e5:.1f} L vs {ext['median_price_main'] / 1e5:.1f} L)")

    ax = axes[1]
    x = np.arange(2); w = 0.3
    for k, key in enumerate(("linear_regression", "random_forest")):
        m = ext["models"][key]
        vals = [m["test_median_ape_main"], m["median_ape"]]
        pos = x + (k - 0.5) * (w + 0.04)
        ax.bar(pos, vals, width=w, color=MODEL_COLORS[key], label=MODEL_NAMES[key], zorder=3)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.008, f"{v:.0%}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x, ["Held-out test set\n(core data)", "External file v4\n(different file, higher prices)"])
    ax.yaxis.set_major_formatter(FuncFormatter(_pct)); ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(0, 0.24); ax.set_ylabel("Median absolute % error"); _clean(ax, "y")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Typical error on a shifted market", pad=28)
    bias = {k: ext["models"][k]["bias_median_signed_pct"] for k in ("linear_regression", "random_forest")}
    note = (f"On v4 the median prediction is {abs(bias['linear_regression']):.0%} (Linear) / {abs(bias['random_forest']):.0%} (Forest) "
            f"{'below' if bias['linear_regression'] < 0 else 'above'} the real price: the models learned 2020 prices.")
    fig.tight_layout(w_pad=3)
    fig.text(0.5, -0.03, note, ha="center", va="top", fontsize=9, color=INK2)
    return fig


def fig_log_vs_raw(res: dict):
    """Same models, same settings; target = log(price) versus target = raw price."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    keys = ["linear_regression", "random_forest"]
    for ax, (metric, title, fmt, axfmt) in zip(axes, [("median_ape", "Median % error (lower is better)", "{:.1%}", _pct),
                                                      ("r2", "R\u00b2 (higher is better)", "{:.2f}", None)]):
        x = np.arange(2); w = 0.3
        for k, (label, color) in enumerate((("raw_price", GRAY), ("log_price", BLUE))):
            vals = [res[label][key][metric] for key in keys]
            pos = x + (k - 0.5) * (w + 0.04)
            ax.bar(pos, vals, width=w, color=color, zorder=3,
                   label="Trained on raw price" if label == "raw_price" else "Trained on log(price)  (our choice)")
            for xi, v in zip(pos, vals):
                ax.text(xi, v + (0.003 if metric == "median_ape" else 0.01), fmt.format(v), ha="center", fontsize=9.5, color=INK)
        ax.set_xticks(x, [MODEL_NAMES[k] for k in keys]); ax.set_title(title, fontsize=10.5); _clean(ax, "y")
        if axfmt:
            ax.yaxis.set_major_formatter(FuncFormatter(axfmt)); ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    axes[0].legend(loc="upper right")
    lin_raw, lin_log = res["raw_price"]["linear_regression"]["median_ape"], res["log_price"]["linear_regression"]["median_ape"]
    rf_raw, rf_log = res["raw_price"]["random_forest"]["median_ape"], res["log_price"]["random_forest"]["median_ape"]
    fig.suptitle(f"Median error with log(price): Linear {lin_raw:.0%} \u2192 {lin_log:.0%}, Random Forest {rf_raw:.0%} \u2192 {rf_log:.0%}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout(w_pad=3)
    return fig
