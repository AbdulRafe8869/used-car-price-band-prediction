"""Used Car Price Predictor  -  live demo (Streamlit).

    streamlit run app/streamlit_app.py

Pick the car's details; the app estimates its selling price in rupees with both
trained models, shows a likely price range, explains which inputs moved the
estimate, and lists comparable real listings from the dataset.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # lets `src` import when run from anywhere
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as cfg
from src import predict as pr
from src import visualize as viz

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

MODEL_LABEL = {"random_forest": "Random Forest", "linear_regression": "Linear Regression"}


# ------------------------------------------------------------- cached loads
@st.cache_resource(show_spinner="Loading models ...")
def get_artifacts():
    return pr.load_artifacts()


@st.cache_data
def get_metrics():
    path = cfg.METRICS_PATH
    return json.loads(path.read_text()) if path.exists() else None


try:
    art = get_artifacts()
except FileNotFoundError:
    st.error("Trained models not found. Run `python -m src.run_all` (or `python -m src.train`) first.")
    st.stop()
meta, data = art["meta"], art["data"]
metrics = get_metrics()
reference = pr.typical_reference(data)


def price_card(price: float) -> str:
    return (f"<div style='background:{viz.SURFACE};border:1px solid {viz.GRID};border-left:6px solid {viz.BLUE};"
            f"border-radius:10px;padding:14px 22px;margin:4px 0 10px 0'>"
            f"<div style='color:{viz.INK2};font-size:.9rem;letter-spacing:.3px'>Estimated selling price</div>"
            f"<div style='color:{viz.INK};font-size:2.6rem;font-weight:700;line-height:1.15'>{pr.inr(price)}</div>"
            f"<div style='color:{viz.INK2};font-size:.95rem'>about Rs {pr.lakh(price)}</div></div>")


def range_chart(price, lo, hi, other_label, other_price, comps_rs):
    """Likely range (bar), the estimate, the other model's estimate and the comparable listings' real prices."""
    fig, ax = plt.subplots(figsize=(6.2, 1.9))
    ax.barh([0], [(hi - lo) / 1e5], left=lo / 1e5, height=0.34, color="#cde2fb", zorder=2)
    if len(comps_rs):
        ax.scatter(np.asarray(comps_rs) / 1e5, np.zeros(len(comps_rs)) - 0.33, s=26, color=viz.MUTED, zorder=3,
                   label="Real comparable listings")
    ax.scatter([other_price / 1e5], [0], s=70, facecolor=viz.SURFACE, edgecolor=viz.ORANGE, linewidth=2.2, zorder=4,
               label=f"{other_label} estimate")
    ax.scatter([price / 1e5], [0], s=95, color=viz.BLUE, edgecolor=viz.SURFACE, linewidth=1.5, zorder=5, label="Estimate")
    ax.axvline(price / 1e5, ymax=0.62, color=viz.BLUE, lw=1, zorder=1)
    ax.text(lo / 1e5, 0.3, f"{lo / 1e5:.1f}", ha="center", va="bottom", fontsize=9, color=viz.INK2)
    ax.text(hi / 1e5, 0.3, f"{hi / 1e5:.1f}", ha="center", va="bottom", fontsize=9, color=viz.INK2)
    ax.set_ylim(-0.6, 0.75); ax.set_yticks([])
    ax.set_xlabel("Price in Rs lakh", fontsize=9, color=viz.INK2)
    ax.tick_params(length=0, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(viz.AXIS)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(pad=0.4)
    return fig


def effect_chart(df: pd.DataFrame):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.barh(range(len(d)), d["effect_rs"] / 1e5, height=0.5,
            color=[viz.BLUE if e >= 0 else viz.ORANGE for e in d["effect_rs"]], zorder=3)
    ax.axvline(0, color=viz.AXIS, lw=1)
    ax.set_yticks(range(len(d)), d["input"]); ax.tick_params(length=0, labelsize=9)
    ax.set_xlabel("Change in the estimated price (Rs lakh)", fontsize=8.5, color=viz.INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(viz.AXIS)
    fig.tight_layout(pad=0.4)
    return fig


# ----------------------------------------------------------------- header
st.title("🚗 Used Car Price Predictor")
st.caption("Machine Learning mini project (CA4) · Linear Regression and Random Forest · CarDekho listings (2020)")

tab_predict, tab_perf, tab_data, tab_about = st.tabs(["Predict", "Model performance", "Data insights", "About"])

# ================================================================= PREDICT
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Car details")
        brand = st.selectbox("Brand", meta["brands"], index=meta["brands"].index("Maruti") if "Maruti" in meta["brands"] else 0)
        options = pr.model_choices(meta, brand)
        labels = [o[0] for o in options] + ["Other / not listed"]
        model_label = st.selectbox("Model", labels, help="Models seen in the training data. Pick 'Other' if yours is missing.")
        model_word = dict(options).get(model_label, "")
        variant = st.text_input("Variant / trim words (optional)", placeholder="e.g. VXI, ZDi, Titanium Diesel, 1.2",
                                help="Words from the listing title after the model name. They help separate base and top variants.")
        c1, c2 = st.columns(2)
        year_min, year_max = meta["year_range"]
        year = c1.slider("Manufacturing year", year_min, year_max, 2015,
                         help="The data covers listings up to 2020, so the model cannot price newer cars.")
        km = c2.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=60_000, step=5_000)
        c3, c4 = st.columns(2)
        fuel_options = [f for f in meta["choices"]["fuel"] if f != "Electric"]   # only 1 electric car in the data
        fuel = c3.selectbox("Fuel", fuel_options, index=fuel_options.index("Petrol"))
        transmission = c4.selectbox("Transmission", meta["choices"]["transmission"], index=meta["choices"]["transmission"].index("Manual"))
        c5, c6 = st.columns(2)
        seller = c5.selectbox("Seller type", meta["choices"]["seller_type"], index=meta["choices"]["seller_type"].index("Individual"))
        owner = c6.selectbox("Owner", meta["choices"]["owner"], index=meta["choices"]["owner"].index("First Owner"))
        if km > 400_000:
            st.warning("Mileage above 400,000 km is extreme; the model caps it at the training-data fence.")

    row = pr.build_input(brand, model_word, variant, year, km, fuel, seller, transmission, owner)
    best_key = (metrics or {}).get("best_model", "random_forest")
    comp = pr.comparable_listings(data, brand, model_word or None, year)
    with right:
        st.subheader("Price estimate")
        choice = st.radio("Model", list(MODEL_LABEL.values()), horizontal=True,
                          index=list(MODEL_LABEL).index(best_key), label_visibility="collapsed")
        key = [k for k, v in MODEL_LABEL.items() if v == choice][0]
        price = pr.predict_price(art[key], row)
        interval = meta["price_interval"][key]
        lo, hi = pr.likely_range(price, interval)
        cover = ((metrics or {}).get("models", {}).get(key, {}).get("likely_range", {}) or {}).get("test_coverage", interval["coverage_target"])
        st.markdown(price_card(price), unsafe_allow_html=True)
        st.markdown(f"**Likely range: {pr.inr(lo)} to {pr.inr(hi)}.** "
                    f"On cars the {choice} had never seen, the real price fell inside a range like this about "
                    f"**{cover:.0%}** of the time.")
        other = [k for k in MODEL_LABEL if k != key][0]
        other_price = pr.predict_price(art[other], row)
        gap = (other_price - price) / price
        if abs(gap) <= 0.10:
            st.success(f"✅ {MODEL_LABEL[other]} agrees: {pr.inr(other_price)} ({gap:+.0%}).")
        else:
            st.info(f"⚖️ {MODEL_LABEL[other]} estimates {pr.inr(other_price)} ({gap:+.0%}). The two models differ, so treat the range as wide.")
        st.pyplot(range_chart(price, lo, hi, MODEL_LABEL[other], other_price, comp["price_lakh"].to_numpy() * 1e5), width="content")

        st.markdown("**Why this price?**")
        eff = pr.explain_by_occlusion(art[key], row, reference)
        st.pyplot(effect_chart(eff), width="content")
        st.caption("Each bar swaps one input for a typical car's value and shows how much the estimate changes. "
                   "Blue = this input raises the price, orange = lowers it.")

    st.subheader("Comparable listings in the dataset")
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={"price_lakh": st.column_config.NumberColumn("Price (Rs lakh)", format="%.2f"),
                                "km_driven": st.column_config.NumberColumn("Km driven", format="%d"),
                                "name": "Listing", "year": st.column_config.NumberColumn("Year", format="%d"),
                                "fuel": "Fuel", "transmission": "Transmission", "owner": "Owner"})
    st.caption(f"Real CarDekho listings (2020) of the same model with the nearest manufacturing year; their median price is "
               f"Rs {comp['price_lakh'].median():.2f} lakh. The estimate is for 2020 market prices; it is a classroom demo, not a valuation.")

# ============================================================ PERFORMANCE
with tab_perf:
    if not metrics:
        st.info("Run `python -m src.evaluate` to create reports/metrics.json.")
    else:
        m = metrics["models"]
        order = ("dummy", "linear_regression", "random_forest")
        st.subheader("Results on the held-out test set")
        st.caption(f"{m['random_forest']['n']} cars the models never saw during training or tuning (80:20 split balanced by price).")
        table = pd.DataFrame({
            "Model": ["Average-price baseline", "Linear Regression", "Random Forest"],
            "R²": [m[k]["r2"] for k in order],
            "Mean error (Rs)": [m[k]["mae_rs"] for k in order],
            "RMSE (Rs)": [m[k]["rmse_rs"] for k in order],
            "Median % error": [m[k]["median_ape"] for k in order],
            "Mean % error": [m[k]["mape"] for k in order],
            "Within ±10 %": [m[k]["within_10pct"] for k in order],
            "Within ±20 %": [m[k]["within_20pct"] for k in order],
        })
        st.dataframe(table.style.format({"R²": "{:.3f}", "Mean error (Rs)": "{:,.0f}", "RMSE (Rs)": "{:,.0f}",
                                         "Median % error": "{:.1%}", "Mean % error": "{:.1%}",
                                         "Within ±10 %": "{:.1%}", "Within ±20 %": "{:.1%}"}),
                     hide_index=True, width="stretch")
        ci = metrics["bootstrap_95ci"]
        st.caption("95% bootstrap intervals: "
                   + " · ".join(f"{MODEL_LABEL[k]} R² {ci[k]['r2'][0]:.3f}–{ci[k]['r2'][1]:.3f}, mean error "
                                f"Rs {ci[k]['mae_rs'][0]:,.0f}–{ci[k]['mae_rs'][1]:,.0f}" for k in MODEL_LABEL))
        for fname, caption in [
            ("07_predicted_vs_actual.png", "Each dot is a test car: the closer to the black line, the better the prediction."),
            ("08_model_comparison.png", None), ("09_error_distribution.png", None),
            ("12_error_by_price_range.png", None), ("10_feature_importance.png", None),
        ]:
            p = cfg.FIGURES_DIR / fname
            if p.exists():
                st.image(str(p), caption=caption, width="stretch")

# ======================================================================= DATA
with tab_data:
    rep = meta["cleaning_report"]
    st.subheader("Dataset: CarDekho used-car listings (Kaggle)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Raw rows", f"{rep['rows_raw']:,}")
    k2.metric("Duplicates removed", f"{rep['duplicates_removed']:,}")
    k3.metric("Clean rows", f"{rep['rows_clean']:,}")
    k4.metric("Train / test", f"{rep['rows_train']:,} / {rep['rows_test']:,}")
    st.dataframe(data.sample(8, random_state=1)[["name", "year", "selling_price", "km_driven", "fuel", "seller_type",
                                                 "transmission", "owner"]], hide_index=True, width="stretch")
    for fname in ["01_price_distribution.png", "02_target_transform.png", "03_price_drivers.png",
                  "04_price_by_brand.png", "05_outliers.png", "06_correlation.png"]:
        p = cfg.FIGURES_DIR / fname
        if p.exists():
            st.image(str(p), width="stretch")

# ====================================================================== ABOUT
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Goal.** Predict the selling price (in rupees) of a used car from its brand, model, age, mileage, fuel, transmission,
seller type and owner history.

**Method.** Duplicate removal and outlier clipping, feature engineering (car age, km per year, brand tier, model and
name keywords), one-hot encoding and scaling inside one scikit-learn pipeline. Both models learn the **logarithm of the
price** (prices are strongly right-skewed) and their answers are converted back to rupees: **Linear Regression**
(with an L2 penalty) and a **Random Forest**, each tuned with Grid Search and 5-fold cross-validation, and compared with an
average-price baseline.

**Likely range.** A price estimate is never exact. The range shown next to it comes from how far real prices were from the
predictions on cars the model had not seen during training, and it was checked again on the test set.

**Limitations.** Prices are 2020 listings, so the demo cannot price newer cars or today's market. Only 7 inputs are used
(no condition, colour, location or accident history), so two cars with the same inputs can still sell for quite different prices.

**Team.** Abdul Rafe (2407070) · Muntasir Ali Khan (2407019) · Shaikh Abdullah Saud (2407040) ·
TY DS, University Dept. of ICT, MGM University · Guide: Ms. Priyanka Hajare.

**Data.** [Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle, nehalbirla).
""")
