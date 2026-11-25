"""Train the Isolation Forest on data/train.csv and save it to models/."""
from __future__ import annotations

import pandas as pd

from . import config
from .model import SyncGuardModel
from .preprocessing import clean


def train_and_save() -> SyncGuardModel:
    raw = pd.read_csv(config.TRAIN_PATH)
    df, info = clean(raw)
    print(f"Training on {len(df):,} rows (dropped {info['rows_dropped']}) - labels are NOT used.")
    model = SyncGuardModel().fit(df)
    path = model.save()
    print(f"Model saved to {path}")
    return model


if __name__ == "__main__":
    train_and_save()
