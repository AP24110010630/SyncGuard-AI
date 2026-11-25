"""Build docs/SyncGuard_AI_Project_Report.docx from the saved evaluation results.

Requires python-docx (pip install python-docx).  Run from the project root:
    python tools/build_report.py
All numbers in the report are read from reports/evaluation_metrics.json so the
document always matches the trained model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src import config

OUT = config.DOCS_DIR / "SyncGuard_AI_Project_Report.docx"
m = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
meta = m["model_meta"]
train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for lvl, size in ((1, 16), (2, 13)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)


def para(text, bold=False, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def table(rows, header, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
        for r in t.rows[0].cells[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def figure(name, caption, width=5.6):
    p = config.CHART_DIR / name
    if p.exists():
        doc.add_picture(str(p), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)


# ------------------------------------------------------------------ title
para("SyncGuard AI", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=28)
para("Pricing / Inventory Sync Anomaly Detector", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
para("Domain: E-Commerce and Retail", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
para("")
para("Project Report", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14)
para("")
for line in ("Student name: ______________________", "Roll number: ______________________",
             "Department: B.Tech Computer Science", "Guide / Faculty: ______________________",
             "Institution: ______________________", "Academic year: ______________________"):
    para(line, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# --------------------------------------------------------------- abstract
doc.add_heading("Abstract", 1)
para(
    "Retailers that sell the same products on several channels (their own web store and marketplaces such as "
    "Amazon or eBay) depend on integrations that keep prices and stock levels synchronised with a master "
    "catalogue. When an integration breaks or lags, customers see wrong prices, items are oversold, or in-stock "
    "products appear sold out. SyncGuard AI learns what normal synchronisation looks like using an unsupervised "
    "Isolation Forest and flags deviations before they reach customers. The system includes a synthetic data "
    "generator, a feature-engineering layer that learns each channel's normal price markup, a rule-based "
    "explanation engine, an evaluation suite, and a Streamlit dashboard with Plotly visualisations, CSV upload "
    f"and CSV export. On a held-out synthetic test set the model reached precision {m['precision']:.2f}, recall "
    f"{m['recall']:.2f} and F1-score {m['f1']:.2f}."
)

doc.add_heading("1. Introduction", 1)
para(
    "Omnichannel retail multiplies the number of places where a price or a stock count can go wrong. Typical "
    "failure modes include a price feed that drops a decimal point, a webhook that stops firing so the channel "
    "keeps selling units that no longer exist, or a nightly job that silently fails. Fixed rules (for example "
    "'alert if the prices differ by more than 5%') are brittle because every channel legitimately differs: "
    "Amazon may carry a fee-driven markup, and stock on a channel is expected to lag behind the master by the "
    "units sold since the last sync. A model that learns these patterns from data can separate harmless "
    "variation from real integration errors."
)

doc.add_heading("2. Problem Statement and Objectives", 1)
para("Problem: detect pricing and inventory synchronisation errors across sales channels, without labelled "
     "examples of failures, early enough to prevent overselling and mispricing.")
bullets([
    "Generate a realistic synthetic dataset with normal and anomalous sync records.",
    "Pre-process the data and engineer features that capture price, inventory and timing behaviour.",
    "Train an unsupervised Isolation Forest and save it for reuse.",
    "Evaluate with precision, recall, F1-score and a confusion matrix.",
    "Provide a professional dashboard with upload, anomaly scores, explanations, charts and CSV export.",
])

doc.add_heading("3. System Architecture", 1)
table(
    [
        ("Data layer", "src/data_generator.py", "Synthetic multi-channel records with injected anomalies; train/test split"),
        ("Pre-processing", "src/preprocessing.py", "Validation, cleaning, feature engineering, per-channel baselines"),
        ("Model layer", "src/model.py, src/train.py", "RobustScaler + Isolation Forest pipeline; save/load; auto-retrain fallback"),
        ("Explanation", "src/explain.py", "Plain-English reason codes with tolerances learned from normal data"),
        ("Evaluation", "src/evaluate.py", "Metrics, confusion matrix, ROC/PR, per-type recall, charts"),
        ("Presentation", "app.py, src/charts.py", "Streamlit dashboard with Plotly charts, upload and download"),
    ],
    ["Layer", "Files", "Responsibility"], [1.3, 1.8, 3.4],
)
para("Data flow: CSV records -> validation and cleaning -> feature engineering -> scaling -> Isolation Forest "
     "score -> threshold (sensitivity) -> flagged records -> explanation engine -> dashboard / CSV export.")

doc.add_heading("4. Dataset", 1)
para(
    f"The generator produces {len(train) + len(test):,} sync observations across 600 SKUs, 7 product categories "
    f"and 5 channels over 60 days. They are split 75/25 (stratified) into {len(train):,} training and "
    f"{len(test):,} test records. The labels (is_anomaly, anomaly_type) exist only for evaluation; the model "
    "never sees them during training."
)
doc.add_heading("4.1 Normal behaviour", 2)
bullets([
    "Channel price = master price x (1 + channel markup) + small noise. Markups differ by channel.",
    "Channel stock = master stock + units sold since the last sync, so the gap grows with sales velocity and delay.",
    "Sync delay is gamma-distributed around about 0.6x the channel's sync cadence.",
    "About 8% of normal records are deliberately 'benign but unusual' (promotions, slow syncs, cart holds) "
    "so the task is not trivially separable.",
])
doc.add_heading("4.2 Injected anomaly types", 2)
tc = train["anomaly_type"].value_counts()
tc2 = test["anomaly_type"].value_counts()
desc = {
    "price_drift": "Channel price drifts 2-35% from the expected channel price",
    "price_decimal_error": "Price multiplied/divided by 10, or set to 0",
    "stock_oversell": "Channel shows 3-100 more units than the master can explain",
    "stock_zeroed": "Channel shows 0 (or negative) stock while master has stock",
    "sync_delay": "Last sync is 3-80x older than the channel cadence",
    "combined": "Price drift + stock gap + long delay together",
    "normal": "Normal records",
}
table([(k, desc.get(k, ""), int(tc.get(k, 0)), int(tc2.get(k, 0))) for k in tc.index],
      ["Type", "Description", "Train", "Test"], [1.8, 3.1, 0.7, 0.7])
doc.add_heading("4.3 Schema", 2)
para("Required columns: " + ", ".join(config.REQUIRED_COLUMNS) + ". Optional: " + ", ".join(config.OPTIONAL_COLUMNS) + ".")

doc.add_heading("5. Pre-processing and Feature Engineering", 1)
para("Column names are normalised, numeric fields are coerced (rows with unusable required values are dropped and "
     "reported), and optional fields receive defaults. A FeatureEngineer object learns, per channel, the median "
     "price difference and its robust spread (MAD) and the usual sync cadence. These baselines are saved with the model.")
table(
    [
        ("price_dev_slog", "Signed log of the price deviation measured in channel-specific robust z-units"),
        ("abs_price_dev_pct", "Absolute deviation from the channel's expected price difference"),
        ("stock_residual_slog", "Signed log of (channel - master stock - expected sales lag)"),
        ("stock_residual_ratio", "Unexplained stock gap relative to master stock"),
        ("log_delay_ratio", "log(1 + sync delay / channel cadence)"),
        ("stockout_mismatch", "1 if channel shows no stock while master has 5+ units"),
        ("invalid_value", "1 if price <= 0 or stock < 0"),
    ],
    ["Model feature", "Meaning"], [1.9, 4.6],
)
para("Signed-log transforms keep extreme errors (such as a x10 price) from dominating the scale, and a RobustScaler "
     "(median / IQR) is applied before the forest.")

doc.add_heading("6. Model: Isolation Forest", 1)
para("Isolation Forest isolates observations by recursively splitting features at random. Anomalies sit in sparse "
     "regions and are isolated in fewer splits, giving them shorter average path lengths and higher anomaly scores. "
     "No labels are needed, which matches reality: most retailers have few labelled examples of sync failures.")
p = meta["params"]
table([("n_estimators", p["n_estimators"]), ("max_samples", p["max_samples"]),
       ("contamination", p["contamination"]), ("random_state", p["random_state"]),
       ("Training rows", f"{meta['n_train_rows']:,}")], ["Parameter", "Value"], [2.5, 2.0])
para("The saved model (models/syncguard_model.joblib) bundles the feature engineer, scaler, forest and explainer. "
     "A sensitivity slider in the dashboard shifts the decision threshold so users can trade precision for recall. "
     "Anomaly scores are also converted to a 0-100 risk score with Critical / High / Medium severity bands.")

doc.add_heading("7. Explainability", 1)
para("The forest says how unusual a record is, not why. The explanation engine compares each flagged record with "
     "tolerances learned from the training records the model itself considers normal:")
tol = meta["tolerances"]
table([("Price deviation", f"{tol['price_deviation_tolerance_pct']}% beyond the channel's normal markup"),
       ("Unexplained stock gap", f"{tol['stock_gap_tolerance_units']} units"),
       ("Sync delay", f"{tol['delay_tolerance_x_cadence']}x the channel cadence")],
      ["Check", "Learned tolerance"], [2.2, 4.0])
para("Learned normal price difference per channel: " +
     ", ".join(f"{k} {v:+.2f}%" for k, v in meta["learned_channel_markup_pct"].items()) + ".")
para("Each flagged record gets one or more reason codes (price mismatch, overselling risk, false stock-out, sync delay, "
     "invalid data) and a category. If no rule applies, the engine reports the features that deviate most.")

doc.add_heading("8. Evaluation", 1)
para(f"Evaluation uses the held-out test set ({m['n_records']:,} records, {m['n_true_anomalies']} true anomalies) at the "
     "default threshold.")
table([("Precision", f"{m['precision']:.3f}"), ("Recall", f"{m['recall']:.3f}"), ("F1-score", f"{m['f1']:.3f}"),
       ("Accuracy", f"{m['accuracy']:.3f}"), ("ROC-AUC", f"{m['roc_auc']:.3f}"), ("PR-AUC", f"{m['pr_auc']:.3f}")],
      ["Metric", "Value"], [2.5, 2.0])
table([("Actual normal", m["tn"], m["fp"]), ("Actual anomaly", m["fn"], m["tp"])],
      ["Confusion matrix", "Predicted normal", "Predicted anomaly"], [2.0, 2.0, 2.0])
doc.add_heading("Recall by anomaly type", 2)
table([(r["anomaly_type"], f"{int(r['detected'])}/{int(r['total'])}", f"{r['recall']:.1%}") for r in m["recall_by_type"]],
      ["Type", "Detected", "Recall"], [2.5, 1.5, 1.5])
para("Obvious failures (decimal errors, false stock-outs, combined faults) are caught reliably. Subtle ones, such as "
     "a 2-3% price drift or an oversell of only a few units, overlap with legitimate promotions and order-timing "
     "noise, which is where recall and precision are lost.")
figure("confusion_matrix.png", "Figure 1: Confusion matrix on the test set", 3.8)
figure("roc_pr_curves.png", "Figure 2: ROC and precision-recall curves")
figure("score_distribution.png", "Figure 3: Anomaly score distribution")
figure("recall_by_anomaly_type.png", "Figure 4: Detection rate by anomaly type")
figure("sensitivity_tradeoff.png", "Figure 5: Precision / recall trade-off against the sensitivity slider")
figure("pca_projection.png", "Figure 6: PCA projection of test records", 4.8)

doc.add_heading("9. Dashboard", 1)
bullets([
    "Data sources: built-in test set, demo sample, or the user's own CSV (validated, with a downloadable template).",
    "KPI cards: records analysed, anomalies flagged, anomaly rate, critical alerts, overselling exposure.",
    "Detected-anomalies tab: filters by channel, category, severity and SKU; anomaly score, 0-100 risk score, "
    "explanations; record inspector; CSV download of filtered or all anomalies.",
    "Visual analysis tabs: pricing differences, inventory mismatches and synchronisation delays (Plotly).",
    "Model performance tab: live precision / recall / F1 / confusion matrix when labels are present, plus saved evaluation charts.",
    "Sensitivity slider to adjust the detection threshold interactively.",
])

doc.add_heading("10. Limitations and Future Work", 1)
bullets([
    "The data is synthetic. Metrics describe performance on this generator, not on a real retailer; real feeds will "
    "need threshold re-tuning and may contain failure modes not simulated here.",
    "The contamination parameter (0.06) matches the synthetic anomaly rate. In production it is unknown; the "
    "sensitivity slider and periodic review of flagged records are the practical substitutes.",
    "Records are scored independently. Time-aware features (rolling per-SKU baselines, change-point detection) would "
    "catch gradual drift.",
    "Explanations are rule-based and complement the model rather than reading its internals; SHAP-style attribution is a possible extension.",
    "Future work: live API connectors (Shopify, Amazon SP-API), alert routing (email / Slack), feedback loop where "
    "reviewers confirm or dismiss alerts and the model is periodically retrained.",
])

doc.add_heading("11. Conclusion", 1)
para("SyncGuard AI demonstrates that an unsupervised model, combined with channel-aware feature engineering and "
     "readable explanations, can surface pricing and inventory synchronisation errors without labelled failure data. "
     "The packaged project reproduces the full pipeline (data, training, evaluation, dashboard) with a single command.")

doc.add_heading("12. Technology Stack", 1)
para("Python, Pandas, NumPy, Scikit-learn (IsolationForest, RobustScaler, Pipeline), Joblib, Matplotlib, Plotly, Streamlit.")

doc.add_heading("References", 1)
bullets([
    "Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. IEEE International Conference on Data Mining (ICDM).",
    "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.",
    "Streamlit documentation: https://docs.streamlit.io",
    "Plotly Python documentation: https://plotly.com/python/",
])

config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("Saved", OUT)
