"""Custom scikit-learn transformers.

Keeping feature engineering and outlier handling *inside* the sklearn Pipeline
means the model takes the same raw columns a user types into the app, and the
exact same steps run in training, cross-validation and prediction.  Nothing is
learned from the test set (no data leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as cfg


def parse_name(name: str) -> tuple[str, str]:
    """Split a CarDekho listing name into (brand, model).

    "Maruti Swift Dzire VDI"      -> ("Maruti", "Maruti Swift")
    "Land Rover Discovery S 2.0"  -> ("Land Rover", "Land Rover Discovery")
    "OpelCorsa 1.4 GL"            -> ("Opel", "Opel Corsa")

    The *model* is the brand plus the first word after it (variants such as
    "VDI" are dropped because they are too sparse to learn from).  Casing is
    normalised so "EcoSport" and "Ecosport" are the same model.
    """
    tokens = str(name).strip().split()
    if not tokens:
        return "Unknown", "Unknown Unknown"
    first = tokens[0]
    if first == "OpelCorsa":                      # two words glued together in the raw data
        brand, rest = "Opel", ["Corsa"] + tokens[1:]
    elif first == "Land" and len(tokens) > 1 and tokens[1].lower() == "rover":
        brand, rest = "Land Rover", tokens[2:]
    else:
        brand, rest = first, tokens[1:]
    model_word = rest[0].title() if rest else "Unknown"
    return brand, f"{brand} {model_word}"


def _keywords(name: str, brand: str) -> str:
    """Listing name without the brand, lower-cased, punctuation turned into spaces.

    "Maruti Swift Dzire VDI" -> "swift dzire vdi"   (the words later become 0/1 features)
    """
    text = str(name).strip()
    for prefix in (brand, "OpelCorsa" if brand == "Opel" else brand):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    for ch in "-()/,+":
        text = text.replace(ch, " ")
    return " ".join(text.lower().split())


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw listing columns -> model-ready engineered columns.

    Input columns : name, year, km_driven, fuel, seller_type, transmission, owner
    Output columns: car_age, km_driven, km_per_year, owner_count,
                    fuel, seller_type, transmission, brand_tier, brand, model,
                    name_keywords   (lower-cased words of the listing name after the brand)
    """

    OUTPUT_COLUMNS = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES + [cfg.TEXT_FEATURE]

    def fit(self, X, y=None):
        self.n_features_in_ = len(cfg.RAW_FEATURES)
        self.feature_names_in_ = np.array(cfg.RAW_FEATURES, dtype=object)
        return self

    def transform(self, X):
        missing = [c for c in cfg.RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = X.reset_index(drop=True)
        parsed = X["name"].map(parse_name)
        brand = parsed.map(lambda t: t[0])
        model = parsed.map(lambda t: t[1])

        out = pd.DataFrame(index=X.index)
        out["car_age"] = (cfg.REFERENCE_YEAR - pd.to_numeric(X["year"], errors="coerce")).clip(lower=0)
        out["km_driven"] = pd.to_numeric(X["km_driven"], errors="coerce")
        out["km_per_year"] = out["km_driven"] / out["car_age"].clip(lower=1)   # usage intensity
        out["owner_count"] = X["owner"].map(cfg.OWNER_MAP)                      # NaN if unseen label
        out["fuel"] = X["fuel"].astype("object")
        out["seller_type"] = X["seller_type"].astype("object")
        out["transmission"] = X["transmission"].astype("object")
        out["brand_tier"] = brand.map(cfg.BRAND_TIER).fillna(cfg.UNKNOWN_TIER)
        out["brand"] = brand
        out["model"] = model
        out[cfg.TEXT_FEATURE] = [_keywords(n, b) for n, b in zip(X["name"], brand)]
        return out[self.OUTPUT_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.OUTPUT_COLUMNS, dtype=object)

    def set_output(self, *, transform=None):  # we always return a DataFrame
        return self


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at Tukey's "far-out" fences.

    lower = Q1 - k*IQR,  upper = Q3 + k*IQR  (k = 3 by default).
    The fences are learned on the *training* data only, then applied to any
    data, so an extreme mileage such as 806,599 km is pulled back to the upper
    fence instead of dragging the scaler and the Logistic Regression weights.
    Rows are never deleted.
    """

    def __init__(self, k: float = cfg.IQR_K):
        self.k = k

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", None)
