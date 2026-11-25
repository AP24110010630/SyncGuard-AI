"""One-command pipeline: generate data -> train model -> evaluate -> save charts.

Usage (from the project root):
    python run_pipeline.py
"""
import json

import pandas as pd

from src import config
from src.data_generator import save_datasets
from src.evaluate import evaluate_model
from src.preprocessing import clean
from src.train import train_and_save


def main():
    print("[1/3] Generating synthetic datasets ...")
    train, test, demo = save_datasets()
    print(f"      train={len(train):,}  test={len(test):,}  demo sample={len(demo):,}")

    print("[2/3] Training Isolation Forest (unsupervised) ...")
    model = train_and_save()

    print("[3/3] Evaluating on the held-out test set ...")
    test_clean, _ = clean(pd.read_csv(config.TEST_PATH))
    m = evaluate_model(model, test_clean)

    print("\n=== Test-set results (default threshold) ===")
    print(m["report_text"])
    print(f"ROC-AUC: {m['roc_auc']:.3f}   PR-AUC: {m['pr_auc']:.3f}")
    print(f"Confusion matrix: TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  TP={m['tp']}")
    print("\nCharts  ->", config.CHART_DIR)
    print("Metrics ->", config.METRICS_PATH)
    print("\nDone. Start the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
