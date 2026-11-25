"""SyncGuard model: feature engineering + RobustScaler + Isolation Forest.

The model is trained WITHOUT labels (unsupervised).  It learns what normal
synchronisation looks like and scores how easily each record is isolated.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from . import config
from .explain import Explainer
from .preprocessing import MODEL_FEATURES, FeatureEngineer, clean

EXTRA_OUTPUT_COLUMNS = [
    "price_diff", "price_diff_pct", "stock_diff", "stock_residual",
    "expected_lag_units", "delay_ratio",
]


def severity_from_risk(risk: float) -> str:
    if risk >= 85:
        return "Critical"
    if risk >= 60:
        return "High"
    return "Medium"


class SyncGuardModel:
    def __init__(self, params: dict | None = None):
        self.params = dict(config.IFOREST_PARAMS if params is None else params)
        self.fe = FeatureEngineer()
        self.pipeline = Pipeline(
            [("scaler", RobustScaler()), ("iforest", IsolationForest(**self.params))]
        )
        self.explainer = Explainer()
        self.risk_lo = 0.4
        self.risk_hi = 0.8
        self.meta: dict = {}

    # -------------------------------------------------------------- training
    def fit(self, df_clean: pd.DataFrame) -> "SyncGuardModel":
        feats = self.fe.fit_transform(df_clean)
        X = feats[MODEL_FEATURES].to_numpy()
        self.pipeline.fit(X)

        raw_scores = -self.pipeline.score_samples(X)        # higher = more anomalous
        self.risk_lo = float(np.percentile(raw_scores, 50))
        self.risk_hi = float(np.percentile(raw_scores, 99.5))

        normal_mask = self.pipeline.decision_function(X) >= 0
        self.explainer = Explainer.from_training(feats[normal_mask])

        self.meta = {
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "n_train_rows": int(len(df_clean)),
            "features": MODEL_FEATURES,
            "params": self.params,
            "tolerances": self.explainer.tolerances(),
            "channels": sorted(self.fe.channel_price_median),
            "learned_channel_markup_pct": {
                k: round(v * 100, 2) for k, v in self.fe.channel_price_median.items()
            },
        }
        return self

    # ------------------------------------------------------------- inference
    @staticmethod
    def threshold_from_sensitivity(sensitivity: float) -> float:
        """Map a 0-100 slider (50 = model default) to a decision threshold."""
        return (float(sensitivity) - 50.0) / 50.0 * 0.10

    def score(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Return features + raw scores (no thresholding)."""
        feats = self.fe.transform(df_clean)
        X = feats[MODEL_FEATURES].to_numpy()
        feats["anomaly_score"] = -self.pipeline.score_samples(X)
        feats["decision_score"] = self.pipeline.decision_function(X)
        return feats

    def predict(self, df_clean: pd.DataFrame, sensitivity: float = 50.0) -> pd.DataFrame:
        """Score, flag and explain every record.  Returns df + result columns."""
        feats = self.score(df_clean)
        threshold = self.threshold_from_sensitivity(sensitivity)
        flagged = feats["decision_score"].to_numpy() < threshold

        span = max(self.risk_hi - self.risk_lo, 1e-6)
        risk = np.clip((feats["anomaly_score"].to_numpy() - self.risk_lo) / span * 100, 0, 100)

        scaled = self.pipeline.named_steps["scaler"].transform(feats[MODEL_FEATURES].to_numpy())
        explanations, categories = self.explainer.explain(
            df_clean, feats, scaled, MODEL_FEATURES, flagged
        )

        out = df_clean.copy()
        out["anomaly_score"] = feats["anomaly_score"].round(4).to_numpy()
        out["risk_score"] = np.round(risk, 1)
        out["predicted_anomaly"] = flagged.astype(int)
        out["severity"] = [severity_from_risk(r) if f else "" for r, f in zip(risk, flagged)]
        out["anomaly_category"] = categories
        out["explanation"] = explanations
        for c in EXTRA_OUTPUT_COLUMNS:
            out[c] = feats[c].round(4).to_numpy()
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path=config.MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path=config.MODEL_PATH) -> "SyncGuardModel":
        return joblib.load(path)


def train_from_csv(path=config.TRAIN_PATH) -> SyncGuardModel:
    df, _ = clean(pd.read_csv(path))
    return SyncGuardModel().fit(df)


def load_or_train(path=config.MODEL_PATH):
    """Load the saved model; retrain automatically if it is missing/incompatible.

    Returns (model, notice).  ``notice`` is None when the saved model loaded
    cleanly, otherwise a short message explaining what happened.
    """
    notice = None
    try:
        model = SyncGuardModel.load(path)
        saved_ver = model.meta.get("sklearn_version")
        if saved_ver and saved_ver.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise RuntimeError(f"scikit-learn version changed ({saved_ver} -> {sklearn.__version__})")
        return model, None
    except Exception as exc:  # missing file, version mismatch, corrupt pickle
        notice = f"Saved model could not be used ({exc}); it was re-trained from data/train.csv."
    if not config.TRAIN_PATH.exists():
        from .data_generator import save_datasets
        save_datasets()
    model = train_from_csv()
    model.save(path)
    return model, notice
