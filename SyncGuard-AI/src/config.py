"""Central configuration for SyncGuard AI.

All paths are resolved relative to the project root, so the project works
from any location (including a folder inside VS Code on Windows).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"
DOCS_DIR = ROOT / "docs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_upload.csv"
SAMPLE_LABELLED_PATH = DATA_DIR / "sample_upload_with_labels.csv"
MODEL_PATH = MODEL_DIR / "syncguard_model.joblib"
METRICS_PATH = REPORT_DIR / "evaluation_metrics.json"

RANDOM_STATE = 42

# --- Sales channel profiles used by the synthetic data generator -------------
# markup      : normal systematic price difference vs the master catalogue
# noise       : random price noise (std, fraction of price)
# sync_freq   : expected minutes between syncs for the channel
CHANNELS = {
    "Shopify":  {"markup": 0.000, "noise": 0.0040, "sync_freq": 15},
    "Amazon":   {"markup": 0.040, "noise": 0.0060, "sync_freq": 30},
    "eBay":     {"markup": 0.020, "noise": 0.0080, "sync_freq": 60},
    "Walmart":  {"markup": -0.015, "noise": 0.0050, "sync_freq": 30},
    "Flipkart": {"markup": 0.010, "noise": 0.0060, "sync_freq": 45},
}

# category -> (mean price, price sigma (log), mean daily sales velocity)
CATEGORIES = {
    "Electronics":    (320.0, 0.80, 4.0),
    "Apparel":        (45.0, 0.50, 9.0),
    "Home & Kitchen": (70.0, 0.60, 6.0),
    "Beauty":         (28.0, 0.45, 12.0),
    "Sports":         (60.0, 0.55, 5.0),
    "Toys":           (25.0, 0.50, 8.0),
    "Grocery":        (9.0, 0.40, 25.0),
}

# --- Input schema ------------------------------------------------------------
REQUIRED_COLUMNS = [
    "sku",
    "channel",
    "master_price",
    "channel_price",
    "master_stock",
    "channel_stock",
    "sync_delay_minutes",
]
OPTIONAL_COLUMNS = [
    "record_id",
    "timestamp",
    "category",
    "sales_velocity_per_day",
    "sync_frequency_minutes",
    "is_anomaly",       # ground-truth label (only used for evaluation)
    "anomaly_type",     # ground-truth type (only used for evaluation)
]
NUMERIC_COLUMNS = [
    "master_price",
    "channel_price",
    "master_stock",
    "channel_stock",
    "sync_delay_minutes",
    "sales_velocity_per_day",
    "sync_frequency_minutes",
]

# --- Model hyper-parameters --------------------------------------------------
IFOREST_PARAMS = {
    "n_estimators": 300,
    "max_samples": 512,
    "contamination": 0.06,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_SYNC_FREQ = 30.0
DEFAULT_VELOCITY = 5.0
