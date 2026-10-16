"""Central configuration for the Used Car Price Prediction project.

Everything that is a *decision* (seeds, column lists, grids, domain maps)
lives here so that the notebook, the training script and the Streamlit app all
use exactly the same values.
"""
from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MAIN_CSV = RAW_DIR / "CAR_DETAILS_FROM_CAR_DEKHO.csv"
V3_CSV = RAW_DIR / "Car_details_v3.csv"
V4_CSV = RAW_DIR / "Car_details_v4.csv"

RF_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression_pipeline.joblib"
DUMMY_MODEL_PATH = MODELS_DIR / "dummy_baseline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# car age reference year and lower sanity bound for the model year
REFERENCE_YEAR = 2020
MIN_YEAR = 1990

# target: the exact selling price in rupees.  Prices are right-skewed (a few luxury
# cars cost 20x the median), so the models learn log(price) and the predictions are
# converted back to rupees (see pipeline.py).
SPLIT_PRICE_QUANTILES = 10          # the 80:20 split keeps the price spread equal in train and test
INTERVAL_COVERAGE = 0.80            # share of cars the "likely price range" is meant to contain

# price ranges used ONLY to analyse where the errors are (never as a prediction target)
ANALYSIS_EDGES = (250_000, 500_000, 1_000_000)
ANALYSIS_LABELS = ("Below Rs 2.5 L", "Rs 2.5 - 5 L", "Rs 5 - 10 L", "Rs 10 L and above")

# columns
TARGET_COL = "selling_price"
RAW_FEATURES = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]

NUMERIC_FEATURES = ["car_age", "km_driven", "km_per_year", "owner_count"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "brand_tier", "brand", "model"]
TEXT_FEATURE = "name_keywords"

# domain maps
OWNER_MAP = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

BRAND_TIER = {
    "Maruti": "Economy", "Tata": "Economy", "Datsun": "Economy",
    "Chevrolet": "Economy", "Fiat": "Economy", "Renault": "Economy",
    "Daewoo": "Economy", "Opel": "Economy", "Ambassador": "Economy",
    "Hyundai": "Mid-range", "Honda": "Mid-range", "Ford": "Mid-range",
    "Toyota": "Mid-range", "Nissan": "Mid-range", "Mahindra": "Mid-range",
    "Volkswagen": "Mid-range", "Skoda": "Mid-range",
    "Mitsubishi": "Mid-range", "Isuzu": "Mid-range", "Force": "Mid-range",
    "Jeep": "Mid-range", "MG": "Mid-range", "Kia": "Mid-range",
    "Audi": "Luxury", "BMW": "Luxury", "Mercedes-Benz": "Luxury",
    "Jaguar": "Luxury", "Land Rover": "Luxury", "Volvo": "Luxury",
}
UNKNOWN_TIER = "Unknown"

MODEL_DISPLAY_ALIASES = {
    "Maruti Wagon": "Maruti Wagon R",
    "Hyundai Grand": "Hyundai Grand i10",
}

# hyper-parameter grids.  Keys are prefixed with the pipeline step "model" and the
# regressor wrapped inside it ("regressor"), e.g. model__regressor__alpha.
LINEAR_GRID = {
    "model__regressor__alpha": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
}
RF_GRID = {
    "model__regressor__n_estimators": [200, 400],
    "model__regressor__max_depth": [None, 25],
    "model__regressor__min_samples_leaf": [1, 2],
    "model__regressor__max_features": [0.2, 0.33, 0.5],
}
LINEAR_GRID_FAST = {"model__regressor__alpha": [1, 10]}
RF_GRID_FAST = {
    "model__regressor__n_estimators": [100],
    "model__regressor__max_depth": [None, 12],
    "model__regressor__max_features": [0.33],
}

IQR_K = 3.0
MIN_CATEGORY_FREQUENCY = 10
APP_MIN_BRAND_LISTINGS = 5
APP_MIN_MODEL_LISTINGS = 2
NAME_KEYWORD_MIN_DF = 10
