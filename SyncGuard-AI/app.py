"""SyncGuard AI - Pricing / Inventory Sync Anomaly Detector (Streamlit dashboard).

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import charts, config
from src.evaluate import compute_metrics, recall_by_type
from src.model import load_or_train
from src.preprocessing import clean, validate_input

st.set_page_config(page_title="SyncGuard AI", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {
        background: rgba(37, 99, 235, 0.07);
        border: 1px solid rgba(37, 99, 235, 0.25);
        padding: 12px 16px; border-radius: 10px;
    }
    .sg-sub {color:#64748b; margin-top:-0.6rem; margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers
def show_chart(fig):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    if fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_or_train()


@st.cache_data(show_spinner=False)
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Scanning records for anomalies ...")
def run_detection(df_raw: pd.DataFrame, sensitivity: float):
    model, _ = get_model()
    df_clean, info = clean(df_raw)
    return model.predict(df_clean, sensitivity=sensitivity), info


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def load_saved_metrics():
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ header
st.title("🛡️ SyncGuard AI")
st.markdown(
    '<p class="sg-sub">Pricing &amp; inventory synchronisation anomaly detector - '
    "unsupervised Isolation Forest for multi-channel retail</p>",
    unsafe_allow_html=True,
)

model, model_notice = get_model()
if model_notice:
    st.warning(model_notice)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose data",
        ["Built-in test dataset (5,000 labelled rows)", "Demo sample (600 rows, no labels)", "Upload my own CSV"],
        label_visibility="collapsed",
    )
    uploaded = None
    if source.startswith("Upload"):
        uploaded = st.file_uploader("CSV file", type=["csv"])
        st.caption("Required columns: " + ", ".join(config.REQUIRED_COLUMNS))
        if config.SAMPLE_PATH.exists():
            st.download_button(
                "Download sample CSV template",
                data=config.SAMPLE_PATH.read_bytes(),
                file_name="syncguard_sample.csv",
                mime="text/csv",
            )

    st.header("Detection settings")
    sensitivity = st.slider(
        "Sensitivity", 0, 100, 50,
        help="50 = model default. Higher flags more records (more recall, more false alarms); "
             "lower flags fewer (higher precision).",
    )
    st.caption("Model: Isolation Forest (unsupervised)")
    st.caption(f"Trained on {model.meta.get('n_train_rows', '?'):,} records")

# ------------------------------------------------------------ load the data
df_raw = None
if source.startswith("Built-in"):
    if config.TEST_PATH.exists():
        df_raw = read_csv_file(str(config.TEST_PATH))
    else:
        st.error("data/test.csv not found. Run `python run_pipeline.py` first.")
elif source.startswith("Demo"):
    if config.SAMPLE_PATH.exists():
        df_raw = read_csv_file(str(config.SAMPLE_PATH))
    else:
        st.error("data/sample_upload.csv not found. Run `python run_pipeline.py` first.")
else:
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to begin. You can download a sample template there.")
    else:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file as CSV: {exc}")
        else:
            ok, msgs = validate_input(df_raw)
            if not ok:
                for m in msgs:
                    st.error(m)
                st.write("Expected columns:")
                st.code(", ".join(config.REQUIRED_COLUMNS + ["(optional) " + c for c in config.OPTIONAL_COLUMNS]))
                df_raw = None

if df_raw is None:
    st.stop()

results, info = run_detection(df_raw, float(sensitivity))
if info["rows_dropped"]:
    st.warning(f"{info['rows_dropped']:,} row(s) were dropped because required numeric fields were missing or invalid.")
if info["defaults_used"]:
    st.caption("Defaults applied: " + "; ".join(info["defaults_used"]))

flagged = results[results["predicted_anomaly"] == 1].sort_values("risk_score", ascending=False)

# --------------------------------------------------------------------- KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Records analysed", f"{len(results):,}")
k2.metric("Anomalies flagged", f"{len(flagged):,}")
k3.metric("Anomaly rate", f"{len(flagged) / max(len(results), 1):.1%}")
k4.metric("Critical alerts", f"{int((flagged['severity'] == 'Critical').sum()):,}")
oversell_units = float(flagged.loc[flagged["stock_residual"] > 0, "stock_residual"].sum()) if len(flagged) else 0.0
k5.metric("Overselling exposure", f"{oversell_units:,.0f} units")

tab_over, tab_anom, tab_price, tab_inv, tab_delay, tab_perf, tab_data = st.tabs(
    ["Overview", "Detected anomalies", "Pricing", "Inventory", "Sync delay", "Model performance", "Data & help"]
)

# ----------------------------------------------------------------- overview
with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.category_bar(results))
    with c2:
        show_chart(charts.channel_bar(results))
    c3, c4 = st.columns(2)
    with c3:
        show_chart(charts.score_hist(results))
    with c4:
        show_chart(charts.severity_donut(results))
    show_chart(charts.timeline(results))

# ---------------------------------------------------------------- anomalies
with tab_anom:
    st.subheader("Detected anomalies")
    if flagged.empty:
        st.success("No anomalies detected at the current sensitivity.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        sel_channels = f1.multiselect("Channel", sorted(flagged["channel"].unique()))
        sel_cats = f2.multiselect("Category", sorted(flagged["anomaly_category"].unique()))
        sel_sev = f3.multiselect("Severity", ["Critical", "High", "Medium"])
        sku_q = f4.text_input("Search SKU")

        view = flagged
        if sel_channels:
            view = view[view["channel"].isin(sel_channels)]
        if sel_cats:
            view = view[view["anomaly_category"].isin(sel_cats)]
        if sel_sev:
            view = view[view["severity"].isin(sel_sev)]
        if sku_q:
            view = view[view["sku"].str.contains(sku_q, case=False, na=False)]

        st.caption(f"Showing {len(view):,} of {len(flagged):,} flagged records, highest risk first.")

        cols = ["record_id", "timestamp", "sku", "channel", "severity", "risk_score", "anomaly_score",
                "anomaly_category", "explanation", "master_price", "channel_price",
                "master_stock", "channel_stock", "sync_delay_minutes"]
        cols = [c for c in cols if c in view.columns]
        show_table(
            view[cols],
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk (0-100)", min_value=0, max_value=100, format="%.0f"),
                "explanation": st.column_config.TextColumn("Explanation", width="large"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download filtered anomalies (CSV)", data=to_csv_bytes(view),
            file_name="syncguard_filtered_anomalies.csv", mime="text/csv", key="dl_filtered",
        )
        d2.download_button(
            "Download all detected anomalies (CSV)", data=to_csv_bytes(flagged),
            file_name="syncguard_detected_anomalies.csv", mime="text/csv", key="dl_all",
        )

        st.divider()
        st.subheader("Inspect a record")
        if len(view):
            options = view["record_id"].head(500).tolist()
            rid = st.selectbox("Record", options)
            row = view[view["record_id"] == rid].iloc[0]
            st.error(f"**{row['severity']} - {row['anomaly_category']}** (risk {row['risk_score']:.0f}/100, "
                     f"anomaly score {row['anomaly_score']:.3f})")
            st.write(row["explanation"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Price (channel vs master)", f"{row['channel_price']:,.2f}", f"{row['channel_price'] - row['master_price']:+,.2f}",
                      delta_color="off")
            m2.metric("Stock (channel vs master)", f"{row['channel_stock']:,.0f}", f"{row['channel_stock'] - row['master_stock']:+,.0f}",
                      delta_color="off")
            m3.metric("Sync delay", f"{row['sync_delay_minutes']:,.0f} min", f"{row['delay_ratio']:.1f}x cadence", delta_color="off")

# -------------------------------------------------------------------- price
with tab_price:
    st.subheader("Pricing differences")
    st.caption("Every channel has its own normal markup (learned from training data), so deviations are measured against that channel's baseline.")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.price_hist(results))
    with c2:
        show_chart(charts.price_box_by_channel(results))
    show_chart(charts.price_scatter(results))

# ---------------------------------------------------------------- inventory
with tab_inv:
    st.subheader("Inventory mismatches")
    st.caption("A small gap is normal (sales since the last sync). The 'unexplained gap' removes that expected lag.")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.stock_gap_hist(results))
    with c2:
        show_chart(charts.stock_direction_bar(results))
    show_chart(charts.stock_scatter(results))

# --------------------------------------------------------------- sync delay
with tab_delay:
    st.subheader("Synchronisation delays")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.delay_hist(results))
    with c2:
        show_chart(charts.delay_box_by_channel(results))
    show_chart(charts.delay_vs_gap(results))

# -------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Model performance")
    if "is_anomaly" in results.columns:
        st.markdown("**Live evaluation on the data currently loaded** (uses its `is_anomaly` column, only for scoring).")
        y_true = results["is_anomaly"].fillna(0).astype(int)
        m = compute_metrics(y_true, results["predicted_anomaly"], results["anomaly_score"])
        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{m['precision']:.3f}")
        b.metric("Recall", f"{m['recall']:.3f}")
        c.metric("F1-score", f"{m['f1']:.3f}")
        d.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(charts.confusion_heatmap(m))
        with c2:
            if "anomaly_type" in results.columns:
                rbt = recall_by_type(results)
                if len(rbt):
                    show_chart(charts.recall_type_bar(rbt))
        st.caption("Move the sensitivity slider in the sidebar to see the precision / recall trade-off change.")
    else:
        st.info("The loaded data has no `is_anomaly` label column, so live metrics are not available. "
                "Saved test-set results from training are shown below.")

    saved = load_saved_metrics()
    st.divider()
    st.markdown("**Saved evaluation from training (held-out test set, default threshold)**")
    if saved:
        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{saved['precision']:.3f}")
        b.metric("Recall", f"{saved['recall']:.3f}")
        c.metric("F1-score", f"{saved['f1']:.3f}")
        d.metric("PR-AUC", f"{saved.get('pr_auc', float('nan')):.3f}")
        imgs = ["confusion_matrix.png", "roc_pr_curves.png", "score_distribution.png",
                "recall_by_anomaly_type.png", "sensitivity_tradeoff.png", "pca_projection.png"]
        cols = st.columns(2)
        for i, name in enumerate(imgs):
            p = config.CHART_DIR / name
            if p.exists():
                cols[i % 2].image(str(p), caption=name.replace("_", " ").replace(".png", "").title())
    else:
        st.info("No saved evaluation found. Run `python run_pipeline.py`.")

    with st.expander("What the model learned"):
        st.write("Normal price difference per channel (learned from training data):")
        st.json(model.meta.get("learned_channel_markup_pct", {}))
        st.write("Explanation tolerances (learned from records the model considers normal):")
        st.json(model.meta.get("tolerances", {}))
        st.write("Model features:")
        st.code(", ".join(model.meta.get("features", [])))

# --------------------------------------------------------------------- data
with tab_data:
    st.subheader("Data preview")
    show_table(results.drop(columns=["explanation"]).head(200))
    st.subheader("CSV format for uploads")
    schema = pd.DataFrame(
        [
            ("sku", "required", "Product identifier"),
            ("channel", "required", "Sales channel name (e.g. Shopify, Amazon)"),
            ("master_price", "required", "Price in the master catalogue / ERP"),
            ("channel_price", "required", "Price currently live on the channel"),
            ("master_stock", "required", "Units in the master inventory"),
            ("channel_stock", "required", "Units shown on the channel"),
            ("sync_delay_minutes", "required", "Minutes since the last successful sync"),
            ("record_id", "optional", "Unique row id (generated if missing)"),
            ("timestamp", "optional", "Observation time (enables the timeline chart)"),
            ("category", "optional", "Product category"),
            ("sales_velocity_per_day", "optional", "Average units sold per day (default 5)"),
            ("sync_frequency_minutes", "optional", "Expected sync interval (learned per channel if missing)"),
            ("is_anomaly", "optional", "0/1 ground truth - enables live precision/recall"),
        ],
        columns=["column", "status", "description"],
    )
    show_table(schema)
