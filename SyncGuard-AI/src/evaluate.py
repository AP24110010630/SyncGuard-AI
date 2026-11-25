"""Model evaluation: precision, recall, F1, confusion matrix and charts.

Ground-truth labels (``is_anomaly``) are used ONLY here, never for training.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from . import config
from .model import SyncGuardModel
from .preprocessing import MODEL_FEATURES

BLUE, RED, GREY = "#2563eb", "#dc2626", "#94a3b8"


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_records": int(len(y_true)),
        "n_true_anomalies": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        m["pr_auc"] = float(average_precision_score(y_true, y_score))
    return m


def recall_by_type(results: pd.DataFrame) -> pd.DataFrame:
    anom = results[results["is_anomaly"] == 1]
    g = anom.groupby("anomaly_type")["predicted_anomaly"].agg(["sum", "count"])
    g.columns = ["detected", "total"]
    g["recall"] = g["detected"] / g["total"]
    return g.sort_values("recall", ascending=False).reset_index()


# ----------------------------------------------------------------- charts
def _save(fig, name):
    config.CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHART_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(m: dict):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black", fontweight="bold")
    return _save(fig, "confusion_matrix.png")


def plot_roc_pr(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    prec, rec, _ = precision_recall_curve(y_true, score)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {roc_auc_score(y_true, score):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=GREY)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color=RED, lw=2, label=f"AP = {average_precision_score(y_true, score):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")
    for a in axes:
        a.grid(alpha=0.25)
    return _save(fig, "roc_pr_curves.png")


def plot_score_distribution(results: pd.DataFrame, threshold_score: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(results["anomaly_score"].min(), results["anomaly_score"].max(), 60)
    ax.hist(results.loc[results["is_anomaly"] == 0, "anomaly_score"], bins=bins, color=BLUE, alpha=0.75, label="Normal")
    ax.hist(results.loc[results["is_anomaly"] == 1, "anomaly_score"], bins=bins, color=RED, alpha=0.75, label="True anomaly")
    ax.axvline(threshold_score, color="black", ls="--", label="Decision threshold")
    ax.set(xlabel="Anomaly score (higher = more anomalous)", ylabel="Records", title="Anomaly score distribution (test set)")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, "score_distribution.png")


def plot_recall_by_type(rbt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(rbt["anomaly_type"], rbt["recall"], color=BLUE)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.set(xlabel="Recall", title="Detection rate by anomaly type (test set)")
    for b, (_, r) in zip(bars, rbt.iterrows()):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{r['recall']:.0%}  ({int(r['detected'])}/{int(r['total'])})", va="center")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "recall_by_anomaly_type.png")


def plot_pca(model: SyncGuardModel, results: pd.DataFrame, df_clean: pd.DataFrame):
    feats = model.fe.transform(df_clean)
    X = model.pipeline.named_steps["scaler"].transform(feats[MODEL_FEATURES].to_numpy())
    # clip so a handful of extreme points do not squash the picture
    X = np.clip(X, -15, 15)
    comp = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    norm = results["is_anomaly"].to_numpy() == 0
    ax.scatter(comp[norm, 0], comp[norm, 1], s=6, c=BLUE, alpha=0.35, label="Normal")
    ax.scatter(comp[~norm, 0], comp[~norm, 1], s=14, c=RED, alpha=0.8, label="True anomaly")
    ax.set(xlabel="PC 1", ylabel="PC 2", title="PCA projection of test records")
    ax.legend()
    ax.grid(alpha=0.2)
    return _save(fig, "pca_projection.png")


def plot_sensitivity_sweep(model: SyncGuardModel, df_clean: pd.DataFrame, y_true):
    feats = model.score(df_clean)
    dec = feats["decision_score"].to_numpy()
    xs = np.linspace(0, 100, 41)
    P, R, F = [], [], []
    for s in xs:
        pred = (dec < model.threshold_from_sensitivity(s)).astype(int)
        P.append(precision_score(y_true, pred, zero_division=0))
        R.append(recall_score(y_true, pred, zero_division=0))
        F.append(f1_score(y_true, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, P, label="Precision", color=BLUE, lw=2)
    ax.plot(xs, R, label="Recall", color=RED, lw=2)
    ax.plot(xs, F, label="F1", color="#16a34a", lw=2, ls="--")
    ax.axvline(50, color=GREY, ls=":", label="Default (50)")
    ax.set(xlabel="Dashboard sensitivity slider", ylabel="Score", title="Precision / recall trade-off vs sensitivity", ylim=(0, 1.02))
    ax.legend()
    ax.grid(alpha=0.25)
    path = _save(fig, "sensitivity_tradeoff.png")
    return path, {"sensitivity": xs.tolist(), "precision": P, "recall": R, "f1": F}


# ------------------------------------------------------------ orchestration
def evaluate_model(model: SyncGuardModel, test_clean: pd.DataFrame) -> dict:
    """Evaluate on a labelled, cleaned test dataframe; write json + charts."""
    results = model.predict(test_clean, sensitivity=50)
    y_true = results["is_anomaly"].astype(int).to_numpy()
    y_pred = results["predicted_anomaly"].to_numpy()
    m = compute_metrics(y_true, y_pred, results["anomaly_score"].to_numpy())

    rbt = recall_by_type(results)
    m["recall_by_type"] = rbt.to_dict(orient="records")
    m["report_text"] = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], digits=3)
    m["model_meta"] = model.meta

    # score at the default threshold (decision_score == 0)
    thr_score = float(results.loc[results["predicted_anomaly"] == 1, "anomaly_score"].min()) if y_pred.any() else 0.5

    plot_confusion(m)
    plot_roc_pr(y_true, results["anomaly_score"].to_numpy())
    plot_score_distribution(results, thr_score)
    plot_recall_by_type(rbt)
    plot_pca(model, results, test_clean)
    _, sweep = plot_sensitivity_sweep(model, test_clean, y_true)
    m["sensitivity_sweep"] = sweep

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=str)
    (config.REPORT_DIR / "classification_report.txt").write_text(m["report_text"], encoding="utf-8")
    results.to_csv(config.REPORT_DIR / "test_predictions.csv", index=False)
    results[results["predicted_anomaly"] == 1].sort_values("risk_score", ascending=False).to_csv(
        config.REPORT_DIR / "test_detected_anomalies.csv", index=False
    )
    return m
