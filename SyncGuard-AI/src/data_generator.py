"""Synthetic multi-channel pricing / inventory synchronisation data.

Each row is one sync observation: the master catalogue value versus the value
currently shown on a sales channel (Shopify, Amazon, ...).

Normal behaviour
----------------
* channel price = master price * (1 + channel markup) + small noise
* channel stock = master stock + units sold since the last sync (the channel
  has not yet deducted them), so the gap grows with sales velocity and delay
* sync delay is roughly gamma distributed around ~0.6x the channel cadence

Injected anomalies (labels are kept ONLY for evaluation, never for training)
---------------------------------------------------------------------------
price_drift, price_decimal_error, stock_oversell, stock_zeroed,
sync_delay, combined
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

ANOMALY_TYPES = {
    "price_drift": 0.22,
    "price_decimal_error": 0.08,
    "stock_oversell": 0.20,
    "stock_zeroed": 0.12,
    "sync_delay": 0.20,
    "combined": 0.18,
}


def _build_catalog(rng: np.random.Generator, n_skus: int) -> pd.DataFrame:
    cats = rng.choice(list(config.CATEGORIES), n_skus)
    base_price = np.empty(n_skus)
    base_vel = np.empty(n_skus)
    for i, c in enumerate(cats):
        mean_p, sigma, mean_v = config.CATEGORIES[c]
        base_price[i] = rng.lognormal(np.log(mean_p), sigma)
        base_vel[i] = rng.gamma(3.0, mean_v / 3.0)
    base_price = np.clip(base_price, 1.5, 3000)
    base_stock = base_vel * rng.uniform(8, 40, n_skus) + 12
    return pd.DataFrame(
        {
            "sku": [f"SKU-{100000 + i}" for i in range(n_skus)],
            "category": cats,
            "base_price": base_price,
            "base_velocity": np.clip(base_vel, 0.3, None),
            "base_stock": base_stock,
        }
    )


def generate_dataset(
    n_records: int = 20000,
    anomaly_rate: float = 0.06,
    seed: int = config.RANDOM_STATE,
    n_skus: int = 600,
    start: str = "2026-01-01",
    days: int = 60,
) -> pd.DataFrame:
    """Create a labelled synthetic dataset of sync observations."""
    rng = np.random.default_rng(seed)
    catalog = _build_catalog(rng, n_skus)

    channel_names = list(config.CHANNELS)
    sku_idx = rng.integers(0, n_skus, n_records)
    chan_idx = rng.integers(0, len(channel_names), n_records)
    cat = catalog.iloc[sku_idx].reset_index(drop=True)
    channel = np.array(channel_names)[chan_idx]

    markup = np.array([config.CHANNELS[c]["markup"] for c in channel])
    noise = np.array([config.CHANNELS[c]["noise"] for c in channel])
    freq = np.array([config.CHANNELS[c]["sync_freq"] for c in channel], dtype=float)

    # Master values: price wiggles slightly over time (promotions / repricing)
    master_price = np.round(cat["base_price"].to_numpy() * (1 + rng.normal(0, 0.02, n_records)), 2)
    master_price = np.clip(master_price, 0.5, None)
    expected_price = master_price * (1 + markup)
    channel_price = expected_price * (1 + rng.normal(0, 1, n_records) * noise)

    master_stock = np.maximum(
        0, np.round(cat["base_stock"].to_numpy() * rng.uniform(0.3, 1.5, n_records))
    ).astype(int)
    velocity = np.round(cat["base_velocity"].to_numpy() * rng.uniform(0.7, 1.3, n_records), 2)

    delay = np.clip(rng.gamma(2.0, freq * 0.30), 0.5, freq * 3.5)
    delay = np.round(delay, 1)
    lag_units = rng.poisson(velocity * delay / 1440.0)
    channel_stock = master_stock + lag_units

    # Benign noise: legitimate behaviour that merely LOOKS unusual (hard negatives)
    benign = rng.random(n_records)
    promo = benign < 0.03                                   # channel-specific promo
    channel_price = np.where(promo, expected_price * rng.uniform(0.94, 0.985, n_records), channel_price)
    slow = (benign >= 0.03) & (benign < 0.06)               # slow-but-legit sync
    delay = np.where(slow, np.round(freq * rng.uniform(2.5, 4.0, n_records), 1), delay)
    channel_stock = np.where(
        slow, master_stock + rng.poisson(velocity * delay / 1440.0), channel_stock
    )
    hold = (benign >= 0.06) & (benign < 0.08)               # cart holds / reservations
    channel_stock = np.where(
        hold, channel_stock + rng.choice([-1, 1], n_records) * rng.integers(3, 12, n_records), channel_stock
    )
    # small random order-timing noise on every record
    channel_stock = channel_stock + rng.integers(-1, 2, n_records)

    ts = pd.to_datetime(start) + pd.to_timedelta(np.sort(rng.integers(0, days * 1440, n_records)), unit="m")

    df = pd.DataFrame(
        {
            "record_id": [f"R{i:07d}" for i in range(n_records)],
            "timestamp": ts,
            "sku": cat["sku"].to_numpy(),
            "category": cat["category"].to_numpy(),
            "channel": channel,
            "master_price": master_price,
            "channel_price": np.round(channel_price, 2),
            "master_stock": master_stock,
            "channel_stock": channel_stock.astype(int),
            "sync_delay_minutes": delay,
            "sales_velocity_per_day": velocity,
            "sync_frequency_minutes": freq,
            "is_anomaly": 0,
            "anomaly_type": "normal",
        }
    )

    # ---- inject anomalies ---------------------------------------------------
    n_anom = int(round(anomaly_rate * n_records))
    anom_idx = rng.choice(n_records, n_anom, replace=False)
    types = rng.choice(list(ANOMALY_TYPES), n_anom, p=list(ANOMALY_TYPES.values()))

    for i, t in zip(anom_idx, types):
        row = df.iloc[i]
        mp, ms, vel, f = row["master_price"], int(row["master_stock"]), row["sales_velocity_per_day"], row["sync_frequency_minutes"]
        exp_p = mp * (1 + config.CHANNELS[row["channel"]]["markup"])
        new_price, new_stock, new_delay = row["channel_price"], int(row["channel_stock"]), row["sync_delay_minutes"]

        def drift():
            return exp_p * (1 + rng.choice([-1, 1]) * float(np.exp(rng.uniform(np.log(0.02), np.log(0.35)))))

        def oversell():
            return ms + int(np.exp(rng.uniform(np.log(3), np.log(100)))) + int(rng.poisson(vel * new_delay / 1440.0))

        if t == "price_drift":
            new_price = drift()
        elif t == "price_decimal_error":
            new_price = rng.choice([exp_p * 10, exp_p / 10, 0.0], p=[0.4, 0.4, 0.2])
        elif t == "stock_oversell":
            new_stock = oversell()
        elif t == "stock_zeroed":
            ms = max(ms, int(rng.integers(10, 60)))
            df.at[df.index[i], "master_stock"] = ms
            new_stock = 0 if rng.random() < 0.8 else -int(rng.integers(1, 20))
        elif t == "sync_delay":
            new_delay = round(f * float(np.exp(rng.uniform(np.log(3), np.log(80)))), 1)
            new_stock = ms + int(rng.poisson(vel * new_delay / 1440.0))
        elif t == "combined":
            new_price = drift()
            new_delay = round(f * float(np.exp(rng.uniform(np.log(3), np.log(50)))), 1)
            new_stock = oversell() + int(rng.poisson(vel * new_delay / 1440.0))

        df.at[df.index[i], "channel_price"] = round(float(new_price), 2)
        df.at[df.index[i], "channel_stock"] = int(new_stock)
        df.at[df.index[i], "sync_delay_minutes"] = float(new_delay)
        df.at[df.index[i], "is_anomaly"] = 1
        df.at[df.index[i], "anomaly_type"] = t

    return df


def generate_and_split(n_records: int = 20000, anomaly_rate: float = 0.06, seed: int = config.RANDOM_STATE):
    """Generate a dataset and return stratified train/test splits (75/25)."""
    from sklearn.model_selection import train_test_split

    df = generate_dataset(n_records, anomaly_rate, seed)
    train, test = train_test_split(
        df, test_size=0.25, stratify=df["is_anomaly"], random_state=seed
    )
    return train.sort_values("timestamp").reset_index(drop=True), test.sort_values("timestamp").reset_index(drop=True)


def save_datasets(n_records: int = 20000, anomaly_rate: float = 0.06):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, test = generate_and_split(n_records, anomaly_rate)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)

    # A separate demo file (different seed / sku catalogue) for the upload feature
    demo = generate_dataset(n_records=600, anomaly_rate=0.08, seed=7, n_skus=120, days=7)
    demo.to_csv(config.SAMPLE_LABELLED_PATH, index=False)
    demo.drop(columns=["is_anomaly", "anomaly_type"]).to_csv(config.SAMPLE_PATH, index=False)
    return train, test, demo


if __name__ == "__main__":
    tr, te, de = save_datasets()
    print(f"train={len(tr)} test={len(te)} demo={len(de)}")
    print(tr["anomaly_type"].value_counts())
