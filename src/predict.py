"""Prediction helpers shared by the Streamlit app, the tests and the notebook.

Kept free of any Streamlit code so it can be unit-tested and reused.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .features import parse_name

INPUT_GROUPS = {                     # human-readable groups of raw inputs, used by the explanation
    "Brand, model and variant": ["name"],
    "Year": ["year"],
    "Km driven": ["km_driven"],
    "Fuel": ["fuel"],
    "Seller type": ["seller_type"],
    "Transmission": ["transmission"],
    "Owner": ["owner"],
}


def load_artifacts() -> dict:
    """Models, app metadata and the cleaned dataset (for comparable listings)."""
    return {
        "random_forest": joblib.load(cfg.RF_MODEL_PATH),
        "linear_regression": joblib.load(cfg.LINEAR_MODEL_PATH),
        "meta": json.loads(cfg.METADATA_PATH.read_text()),
        "data": pd.read_csv(cfg.PROCESSED_DIR / "cleaned_data.csv"),
    }


# ------------------------------------------------------------- formatting
def inr(amount: float, round_to: int = 1000) -> str:
    """Indian digit grouping, rounded: 462300 -> 'Rs 4,62,000'."""
    n = int(round(float(amount) / round_to) * round_to)
    s = str(abs(n))
    head, tail = (s[:-3], s[-3:]) if len(s) > 3 else ("", s)
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    body = ",".join(groups + [tail])
    return f"Rs {'-' if n < 0 else ''}{body}"


def lakh(amount: float) -> str:
    """462300 -> '4.62 lakh'."""
    return f"{float(amount) / 1e5:.2f} lakh"


# ------------------------------------------------------------ prediction
def compose_name(brand: str, model: str | None, variant: str = "") -> str:
    """Brand + model + optional variant words, in the CarDekho listing-name style."""
    parts = [brand, (model or "").strip(), (variant or "").strip()]
    return " ".join(p for p in parts if p)


def build_input(brand, model, variant, year, km_driven, fuel, seller_type, transmission, owner) -> pd.DataFrame:
    """One-row DataFrame with exactly the raw columns the pipeline expects."""
    return pd.DataFrame([{
        "name": compose_name(brand, model, variant),
        "year": int(year), "km_driven": float(km_driven),
        "fuel": fuel, "seller_type": seller_type, "transmission": transmission, "owner": owner,
    }])[cfg.RAW_FEATURES]


def predict_price(pipeline, row: pd.DataFrame) -> float:
    """Estimated selling price in rupees."""
    return float(pipeline.predict(row)[0])


def likely_range(price: float, interval: dict) -> tuple[float, float]:
    """The 'likely price range': about 80 % of similar cars sold inside it (checked on the test set)."""
    return price * interval["lower_multiplier"], price * interval["upper_multiplier"]


def typical_reference(data: pd.DataFrame) -> pd.Series:
    """A 'typical car' (median year / km, most common categories) used as the neutral baseline."""
    names = data["name"].map(lambda n: parse_name(n)[1])
    return pd.Series({
        "name": names.mode().iloc[0],
        "year": int(data["year"].median()), "km_driven": float(data["km_driven"].median()),
        "fuel": data["fuel"].mode().iloc[0], "seller_type": data["seller_type"].mode().iloc[0],
        "transmission": data["transmission"].mode().iloc[0], "owner": data["owner"].mode().iloc[0],
    })


def explain_by_occlusion(pipeline, row: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """How many rupees does each input add to (or take off) the estimate?

    For every input group we swap *only that input* for the typical value and see
    how the estimated price changes.  A positive number means the car's own value
    makes it MORE expensive than a typical car's value would.
    (A simple, model-agnostic "what if this were ordinary?" explanation.)
    """
    base = predict_price(pipeline, row)
    rows = []
    for group, cols in INPUT_GROUPS.items():
        alt = row.copy()
        for c in cols:
            alt[c] = reference[c]
        rows.append({"input": group, "effect_rs": base - predict_price(pipeline, alt)})
    return pd.DataFrame(rows).sort_values("effect_rs", key=np.abs, ascending=False).reset_index(drop=True)


def comparable_listings(data: pd.DataFrame, brand: str, model: str | None, year: int, n: int = 8) -> pd.DataFrame:
    """Real listings of the same model (or brand) with the closest manufacturing year."""
    parsed = data["name"].map(parse_name)
    same_brand = data[parsed.map(lambda t: t[0]) == brand].copy()
    if model:
        key = f"{brand} {model}".strip()
        same_model = same_brand[parsed.loc[same_brand.index].map(lambda t: t[1]) == key]
        pool = same_model if len(same_model) >= 3 else same_brand
    else:
        pool = same_brand
    pool = pool.assign(year_gap=(pool["year"] - year).abs()).sort_values(["year_gap", "km_driven"]).head(n)
    out = pool[["name", "year", "km_driven", "fuel", "transmission", "owner", cfg.TARGET_COL]].copy()
    out[cfg.TARGET_COL] = (out[cfg.TARGET_COL] / 1e5).round(2)
    return out.rename(columns={cfg.TARGET_COL: "price_lakh"}).reset_index(drop=True)


def model_choices(meta: dict, brand: str) -> list[tuple[str, str]]:
    """[(label shown in the app, model word used by the pipeline), ...]."""
    out = []
    for m in meta["models_by_brand"].get(brand, []):
        key = f"{brand} {m}"
        out.append((cfg.MODEL_DISPLAY_ALIASES.get(key, key)[len(brand) + 1:], m))
    return sorted(out)
