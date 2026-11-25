"""Automated checks. Run from the project root:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_generator import generate_dataset
from src.evaluate import compute_metrics
from src.model import SyncGuardModel, load_or_train
from src.preprocessing import MODEL_FEATURES, clean, validate_input


class TestData(unittest.TestCase):
    def test_generator_schema_and_rate(self):
        df = generate_dataset(n_records=2000, anomaly_rate=0.06, seed=1)
        for col in config.REQUIRED_COLUMNS + ["is_anomaly", "anomaly_type"]:
            self.assertIn(col, df.columns)
        self.assertEqual(int(df["is_anomaly"].sum()), 120)
        self.assertFalse(df[config.REQUIRED_COLUMNS].isna().any().any())

    def test_dataset_files_exist(self):
        for p in (config.TRAIN_PATH, config.TEST_PATH, config.SAMPLE_PATH, config.MODEL_PATH):
            self.assertTrue(p.exists(), f"missing {p}")

    def test_validation_flags_missing_columns(self):
        ok, msgs = validate_input(pd.DataFrame({"sku": ["a"], "channel": ["Shopify"]}))
        self.assertFalse(ok)
        self.assertIn("master_price", msgs[0])


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, notice = load_or_train()
        cls.test_df, _ = clean(pd.read_csv(config.TEST_PATH))

    def test_features_finite(self):
        feats = self.model.fe.transform(self.test_df)
        self.assertTrue(np.isfinite(feats[MODEL_FEATURES].to_numpy()).all())

    def test_quality_on_holdout(self):
        res = self.model.predict(self.test_df)
        m = compute_metrics(res["is_anomaly"], res["predicted_anomaly"], res["anomaly_score"])
        self.assertGreater(m["f1"], 0.70)
        self.assertGreater(m["roc_auc"], 0.95)

    def test_flagged_rows_are_explained(self):
        res = self.model.predict(self.test_df)
        fl = res[res["predicted_anomaly"] == 1]
        self.assertGreater(len(fl), 0)
        self.assertTrue((fl["explanation"].str.len() > 0).all())
        self.assertTrue((fl["severity"] != "").all())
        self.assertTrue((res.loc[res["predicted_anomaly"] == 0, "explanation"] == "").all())

    def test_sensitivity_is_monotonic(self):
        lo = self.model.predict(self.test_df, 20)["predicted_anomaly"].sum()
        mid = self.model.predict(self.test_df, 50)["predicted_anomaly"].sum()
        hi = self.model.predict(self.test_df, 80)["predicted_anomaly"].sum()
        self.assertLessEqual(lo, mid)
        self.assertLessEqual(mid, hi)

    def test_upload_with_only_required_columns_and_unknown_channel(self):
        raw = pd.DataFrame({
            "SKU": ["A1", "A2", "A3"],
            "Channel": ["Shopify", "NewMarket", "Amazon"],
            "Master Price": [10.0, 20.0, 30.0],
            "Channel Price": [10.0, 200.0, 31.2],
            "Master Stock": [50, 40, 30],
            "Channel Stock": [50, 40, 0],
            "Sync Delay Minutes": [10, 20, 5000],
        })
        ok, _ = validate_input(raw)
        self.assertTrue(ok)
        df, info = clean(raw)
        res = self.model.predict(df)
        self.assertEqual(len(res), 3)
        self.assertEqual(res["predicted_anomaly"].tolist()[1:], [1, 1])

    def test_bad_rows_are_dropped_not_crashed(self):
        raw = pd.DataFrame({
            "sku": ["A", "B"], "channel": ["Shopify", "Shopify"],
            "master_price": [10, "oops"], "channel_price": [10, 10],
            "master_stock": [5, 5], "channel_stock": [5, 5], "sync_delay_minutes": [5, 5],
        })
        df, info = clean(raw)
        self.assertEqual(info["rows_dropped"], 1)
        self.assertEqual(len(self.model.predict(df)), 1)

    def test_retrains_when_model_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.joblib"
            model, notice = load_or_train(path)
            self.assertIsNotNone(notice)
            self.assertTrue(path.exists())
            self.assertIsInstance(model, SyncGuardModel)


if __name__ == "__main__":
    unittest.main()
