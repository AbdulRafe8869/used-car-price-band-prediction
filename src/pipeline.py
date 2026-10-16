"""Pipeline factories: preprocessing + the two regressors + a baseline.

Pipeline = FeatureEngineer -> ColumnTransformer -> regressor

    numeric    : impute (median) -> clip outliers (IQR fences) -> StandardScaler
    categorical: impute (most frequent) -> OneHotEncoder (rare levels pooled)
    keywords   : words of the listing name -> 0/1 presence flags (CountVectorizer)

The regressor learns log(price) and its answer is converted back to rupees
(TransformedTargetRegressor).  Prices are right-skewed, so on the log scale a
mistake of 10 % counts the same for a Rs 1.5 lakh hatchback and a Rs 30 lakh SUV.
Predictions therefore always come out in plain rupees.

Because every step lives in one Pipeline, cross-validation re-fits the imputer,
clipper, scaler and encoder on each training fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .features import FeatureEngineer, IQRClipper


def build_preprocessor(numeric=None, categorical=None, use_keywords=True) -> ColumnTransformer:
    numeric = cfg.NUMERIC_FEATURES if numeric is None else numeric
    categorical = cfg.CATEGORICAL_FEATURES if categorical is None else categorical
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", IQRClipper(k=cfg.IQR_K)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=cfg.MIN_CATEGORY_FREQUENCY,
            sparse_output=False,
        )),
    ])
    blocks = []
    if numeric:
        blocks.append(("num", numeric_steps, list(numeric)))
    if categorical:
        blocks.append(("cat", categorical_steps, list(categorical)))
    if use_keywords:
        blocks.append(("kw", CountVectorizer(
            min_df=cfg.NAME_KEYWORD_MIN_DF, binary=True, token_pattern=r"[a-z0-9][a-z0-9\.]*",
        ), cfg.TEXT_FEATURE))          # scalar column name -> 1-D text input
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)


def log_price(regressor) -> TransformedTargetRegressor:
    """Wrap a regressor so it is trained on log(price) and predicts rupees."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_pipeline(estimator, numeric=None, categorical=None, use_keywords=True) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("prep", build_preprocessor(numeric, categorical, use_keywords)),
        ("model", estimator),
    ])


def make_dummy() -> Pipeline:
    """Baseline: always predict the average training price.  The floor any real model must beat."""
    return build_pipeline(DummyRegressor(strategy="mean"))


def make_linear_regression(**kw) -> Pipeline:
    """Linear Regression with an L2 penalty (Ridge) on log(price)."""
    params = dict(alpha=1.0, random_state=cfg.RANDOM_STATE)
    params.update(kw)
    return build_pipeline(log_price(Ridge(**params)))


def make_random_forest(**kw) -> Pipeline:
    params = dict(n_estimators=300, max_features=0.33, random_state=cfg.RANDOM_STATE, n_jobs=1)
    params.update(kw)
    return build_pipeline(log_price(RandomForestRegressor(**params)))


def final_estimator(pipeline: Pipeline):
    """The fitted regressor itself (unwrapping the log-price wrapper if present)."""
    est = pipeline.named_steps["model"]
    return getattr(est, "regressor_", est)


def transformed_feature_names(pipeline: Pipeline):
    """Names of the columns the regressor actually sees (after one-hot etc.)."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())
