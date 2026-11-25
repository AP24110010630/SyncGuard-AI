"""Data validation, cleaning and feature engineering.

The :class:`FeatureEngineer` learns the *normal* per-channel price relationship
(each channel has its own systematic markup) from the training data and uses it
to turn raw sync records into model features.  Only plain Python dicts/floats
are stored on the object so that the saved model stays portable between
pandas versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MODEL_FEATURES = [
    "price_dev_slog",       # channel-adjusted price deviation (signed log)
    "abs_price_dev_pct",    # absolute deviation from expected channel price
    "stock_residual_slog",  # unexplained stock gap (signed log)
    "stock_residual_ratio", # stock gap relative to master stock
    "log_delay_ratio",      # sync delay relative to channel cadence
    "stockout_mismatch",    # channel out of stock while master has stock
    "invalid_value",        # zero/negative price or negative stock
]

FEATURE_LABELS = {
    "price_dev_slog": "price deviation",
    "abs_price_dev_pct": "price deviation",
    "stock_residual_slog": "stock gap",
    "stock_residual_ratio": "stock gap",
    "log_delay_ratio": "sync delay",
    "stockout_mismatch": "false stock-out",
    "invalid_value": "invalid value",
}


def slog(x):
    """Signed log transform: compresses huge outliers, keeps the sign."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def validate_input(df: pd.DataFrame):
    """Return (is_valid, list_of_messages) for an uploaded dataframe."""
    cols = set(normalise_columns(df).columns)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in cols]
    msgs = []
    if missing:
        msgs.append("Missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        msgs.append("The file contains no rows.")
    return (not msgs), msgs


def clean(df: pd.DataFrame):
    """Standardise column names / types and fill optional columns.

    Returns (clean_df, info) where info reports rows dropped and defaults used.
    """
    out = normalise_columns(df)
    info = {"rows_in": len(out), "rows_dropped": 0, "defaults_used": []}

    for c in config.NUMERIC_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["sku"] = out["sku"].astype(str).str.strip()
    out["channel"] = out["channel"].astype(str).str.strip()

    before = len(out)
    required_numeric = [c for c in config.REQUIRED_COLUMNS if c in config.NUMERIC_COLUMNS]
    out = out.dropna(subset=required_numeric).copy()
    info["rows_dropped"] = before - len(out)

    if "record_id" not in out.columns:
        out["record_id"] = [f"ROW-{i + 1:06d}" for i in range(len(out))]
        info["defaults_used"].append("record_id generated")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "category" not in out.columns:
        out["category"] = "Unknown"
    if "sales_velocity_per_day" not in out.columns:
        out["sales_velocity_per_day"] = config.DEFAULT_VELOCITY
        info["defaults_used"].append(f"sales_velocity_per_day = {config.DEFAULT_VELOCITY}")
    out["sales_velocity_per_day"] = out["sales_velocity_per_day"].fillna(config.DEFAULT_VELOCITY).clip(lower=0)
    if "sync_frequency_minutes" not in out.columns:
        out["sync_frequency_minutes"] = np.nan
        info["defaults_used"].append("sync_frequency_minutes learned per channel")
    return out.reset_index(drop=True), info


class FeatureEngineer:
    """Learns channel baselines on training data and builds model features."""

    def __init__(self):
        self.channel_price_median: dict[str, float] = {}
        self.channel_price_mad: dict[str, float] = {}
        self.channel_sync_freq: dict[str, float] = {}
        self.global_price_median = 0.0
        self.global_price_mad = 0.01
        self.global_sync_freq = config.DEFAULT_SYNC_FREQ
        self.fitted = False

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _price_pct(df: pd.DataFrame) -> np.ndarray:
        mp = df["master_price"].to_numpy(dtype=float)
        cp = df["channel_price"].to_numpy(dtype=float)
        return np.where(mp > 0, (cp - mp) / np.where(mp > 0, mp, 1.0), 0.0)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        pct = self._price_pct(df)
        self.global_price_median = float(np.median(pct))
        self.global_price_mad = float(max(1.4826 * np.median(np.abs(pct - self.global_price_median)), 0.003))
        freq_col = df["sync_frequency_minutes"].to_numpy(dtype=float)
        if np.isfinite(freq_col).any():
            self.global_sync_freq = float(np.nanmedian(freq_col))
        for ch, idx in df.groupby("channel").groups.items():
            pos = df.index.get_indexer(idx)
            p = pct[pos]
            med = float(np.median(p))
            mad = float(max(1.4826 * np.median(np.abs(p - med)), 0.003))
            self.channel_price_median[str(ch)] = med
            self.channel_price_mad[str(ch)] = mad
            f = freq_col[pos]
            self.channel_sync_freq[str(ch)] = (
                float(np.nanmedian(f)) if np.isfinite(f).any() else self.global_sync_freq
            )
        self.fitted = True
        return self

    # ------------------------------------------------------------ transform
    def _lookup(self, channels: pd.Series, table: dict, default: float) -> np.ndarray:
        return channels.map(table).astype(float).fillna(default).to_numpy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with all engineered columns (model + explanation)."""
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")
        mp = df["master_price"].to_numpy(dtype=float)
        cp = df["channel_price"].to_numpy(dtype=float)
        ms = df["master_stock"].to_numpy(dtype=float)
        cs = df["channel_stock"].to_numpy(dtype=float)
        delay = df["sync_delay_minutes"].to_numpy(dtype=float)
        vel = df["sales_velocity_per_day"].to_numpy(dtype=float)

        ch = df["channel"].astype(str)
        exp_pct = self._lookup(ch, self.channel_price_median, self.global_price_median)
        mad = self._lookup(ch, self.channel_price_mad, self.global_price_mad)
        freq = df["sync_frequency_minutes"].to_numpy(dtype=float)
        learned_freq = self._lookup(ch, self.channel_sync_freq, self.global_sync_freq)
        freq = np.where(np.isfinite(freq) & (freq > 0), freq, learned_freq)

        price_diff = cp - mp
        price_diff_pct = self._price_pct(df)
        dev = price_diff_pct - exp_pct
        price_dev_z = dev / mad

        stock_diff = cs - ms
        expected_lag = vel * delay / 1440.0
        stock_residual = stock_diff - expected_lag
        stock_residual_ratio = np.clip(stock_residual / (np.clip(ms, 0, None) + 10.0), -5, 5)

        delay_ratio = delay / freq
        feats = pd.DataFrame(
            {
                "price_diff": price_diff,
                "price_diff_pct": price_diff_pct,
                "expected_price_diff_pct": exp_pct,
                "price_dev_z": price_dev_z,
                "price_dev_slog": slog(price_dev_z),
                "abs_price_dev_pct": np.clip(np.abs(dev), 0, 5),
                "stock_diff": stock_diff,
                "expected_lag_units": expected_lag,
                "stock_residual": stock_residual,
                "stock_residual_slog": slog(stock_residual),
                "stock_residual_ratio": stock_residual_ratio,
                "delay_ratio": delay_ratio,
                "expected_sync_minutes": freq,
                "log_delay_ratio": np.log1p(np.clip(delay_ratio, 0, None)),
                "stockout_mismatch": ((cs <= 0) & (ms >= 5)).astype(int),
                "invalid_value": ((cp <= 0) | (mp <= 0) | (cs < 0) | (ms < 0)).astype(int),
            },
            index=df.index,
        )
        return feats

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
