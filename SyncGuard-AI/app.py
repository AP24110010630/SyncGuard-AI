"""SyncGuard AI - Pricing / Inventory Sync Anomaly Detector (Streamlit dashboard).

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import charts, config, ui
from src.evaluate import compute_metrics, recall_by_type
from src.model import load_or_train
from src.preprocessing import clean, normalise_columns, validate_input

st.set_page_config(
    page_title="SyncGuard AI | E-Commerce Sync Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_global_styles()


# ----------------------------------------------------------------- helpers
def show_chart(fig) -> None:
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    if fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": "hover", "displaylogo": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_download_button(*args, **kwargs) -> None:
    try:
        st.download_button(*args, width="stretch", **kwargs)
    except TypeError:
        st.download_button(*args, use_container_width=True, **kwargs)


@st.cache_resource(show_spinner="Loading Isolation Forest model ...")
def get_model():
    return load_or_train()


@st.cache_data(show_spinner=False)
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Scanning records for sync anomalies ...")
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


def _store_uploaded_file(uploaded_file) -> None:
    if uploaded_file is not None:
        st.session_state["uploaded_csv_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_csv_name"] = uploaded_file.name


def _read_uploaded_df():
    raw_bytes = st.session_state.get("uploaded_csv_bytes")
    if not raw_bytes:
        return None, None
    return pd.read_csv(io.BytesIO(raw_bytes)), st.session_state.get("uploaded_csv_name", "uploaded.csv")


def _schema_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
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
        columns=["Column", "Status", "Description"],
    )


# ------------------------------------------------------------------- model
model, model_notice = get_model()

PAGES = [
    "Overview",
    "Detected Anomalies",
    "Pricing Analysis",
    "Inventory Analysis",
    "Sync Delay Analysis",
    "Model Performance",
    "Upload & Schema",
]

PAGE_META = {
    "Overview": (
        "Monitoring Overview",
        "Real-time health across pricing, inventory, and channel synchronisation pipelines.",
    ),
    "Detected Anomalies": (
        "Detected Anomalies",
        "Filter flagged records by channel, category, severity, and risk score, then inspect root causes.",
    ),
    "Pricing Analysis": (
        "Pricing Analysis",
        "Compare live channel prices against master catalogue prices and per-channel markup baselines.",
    ),
    "Inventory Analysis": (
        "Inventory Analysis",
        "Track overselling exposure, false stock-outs, and unexplained stock gaps after expected sales lag.",
    ),
    "Sync Delay Analysis": (
        "Synchronisation Delay Analysis",
        "Measure sync latency relative to each channel's normal cadence and spot stalled integrations.",
    ),
    "Model Performance": (
        "Model Performance",
        "Evaluate Isolation Forest precision, recall, F1, sensitivity trade-offs, and learned baselines.",
    ),
    "Upload & Schema": (
        "Upload & Data Schema",
        "Upload your own catalogue sync CSV, validate required columns, and preview scanned records.",
    ),
}

DATA_SOURCES = [
    "Built-in test dataset (5,000 labelled rows)",
    "Demo sample (600 rows, no labels)",
    "Upload my own CSV",
]

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    ui.render_sidebar_brand()

    ui.render_sidebar_section_label("Navigation")
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="nav_page",
    )

    ui.render_sidebar_section_label("Data Source")
    source = st.radio(
        "Choose data source",
        DATA_SOURCES,
        label_visibility="collapsed",
        key="data_source",
    )

    if source.startswith("Upload"):
        sidebar_upload = st.file_uploader("Upload CSV file", type=["csv"], key="sidebar_uploader")
        _store_uploaded_file(sidebar_upload)
        st.caption("Required: " + ", ".join(config.REQUIRED_COLUMNS))
        if config.SAMPLE_PATH.exists():
            show_download_button(
                "Download sample CSV template",
                data=config.SAMPLE_PATH.read_bytes(),
                file_name="syncguard_sample.csv",
                mime="text/csv",
            )

    ui.render_sidebar_section_label("Detection Settings")
    sensitivity = st.slider(
        "Sensitivity",
        0,
        100,
        50,
        help=(
            "50 = model default. Higher flags more records (more recall, more false alarms); "
            "lower flags fewer (higher precision)."
        ),
    )

    ui.render_sidebar_model_card(
        model.meta.get("n_train_rows", "?"),
        model.meta.get("channels", []),
    )


# ------------------------------------------------------------ load the data
df_raw = None
dataset_label = "Built-in test (5,000)"
upload_error = None
upload_cols_present: set[str] | None = None

if source.startswith("Built-in"):
    dataset_label = "Test set (5,000)"
    if config.TEST_PATH.exists():
        df_raw = read_csv_file(str(config.TEST_PATH))
    else:
        st.error("data/test.csv not found. Run `python run_pipeline.py` first.")
elif source.startswith("Demo"):
    dataset_label = "Demo sample (600)"
    if config.SAMPLE_PATH.exists():
        df_raw = read_csv_file(str(config.SAMPLE_PATH))
    else:
        st.error("data/sample_upload.csv not found. Run `python run_pipeline.py` first.")
else:
    dataset_label = st.session_state.get("uploaded_csv_name", "Custom CSV")
    try:
        loaded_df, loaded_name = _read_uploaded_df()
        if loaded_df is not None:
            upload_cols_present = set(normalise_columns(loaded_df).columns)
            ok, msgs = validate_input(loaded_df)
            if ok:
                df_raw = loaded_df
                dataset_label = f"{loaded_name} ({len(df_raw):,})"
            else:
                upload_error = msgs
    except Exception as exc:
        upload_error = [f"Could not read the file as CSV: {exc}"]

if model_notice:
    st.warning(model_notice)

# Handle upload-empty or invalid state gracefully with a polished upload card
if df_raw is None:
    title, subtitle = PAGE_META[page]
    ui.render_top_bar(title, subtitle, dataset_label, sensitivity=sensitivity)
    with st.container(border=True):
        ui.render_section_header(
            "Upload your sync CSV to start scanning",
            "Drag and drop a CSV export from your ERP or channel integrator, or download the sample template.",
        )
        main_upload = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="empty_state_uploader",
        )
        if main_upload is not None:
            _store_uploaded_file(main_upload)
            st.rerun()

        if upload_error:
            for msg in upload_error:
                st.error(msg)
        if upload_cols_present is not None:
            ui.render_validation_checklist(upload_cols_present, config.REQUIRED_COLUMNS, config.OPTIONAL_COLUMNS)

        c_dl, c_sp = st.columns([1, 2])
        if config.SAMPLE_PATH.exists():
            c_dl.download_button(
                "Download sample CSV template",
                data=config.SAMPLE_PATH.read_bytes(),
                file_name="syncguard_sample.csv",
                mime="text/csv",
                key="empty_dl_sample",
            )
    with st.container(border=True):
        ui.render_section_header("Expected CSV Schema", "Column names are case-insensitive and spaces are treated as underscores.")
        show_table(_schema_dataframe())
    st.stop()


# ----------------------------------------------------------- run detection
results, info = run_detection(df_raw, float(sensitivity))
flagged = results[results["predicted_anomaly"] == 1].sort_values("risk_score", ascending=False)

title, subtitle = PAGE_META[page]
ui.render_top_bar(
    title,
    subtitle,
    dataset_label,
    n_records=len(results),
    n_flagged=len(flagged),
    sensitivity=sensitivity,
)

if info["rows_dropped"]:
    st.warning(f"{info['rows_dropped']:,} row(s) were dropped because required numeric fields were missing or invalid.")
if info["defaults_used"]:
    st.caption("Defaults applied: " + "; ".join(info["defaults_used"]))


# =================================================================== PAGES
if page == "Overview":
    oversell_units = float(flagged.loc[flagged["stock_residual"] > 0, "stock_residual"].sum()) if len(flagged) else 0.0
    critical_count = int((flagged["severity"] == "Critical").sum())
    anomaly_rate = len(flagged) / max(len(results), 1)

    ui.render_kpi_row(
        [
            {
                "label": "Records Analysed",
                "value": f"{len(results):,}",
                "subtitle": f"Across {results['channel'].nunique()} sales channels",
                "accent": "#2563EB",
                "badge": "Active",
            },
            {
                "label": "Anomalies Flagged",
                "value": f"{len(flagged):,}",
                "subtitle": f"At sensitivity {sensitivity}/100",
                "accent": "#DC2626" if len(flagged) else "#16A34A",
                "badge": "Detected",
                "badge_bg": "#FEF2F2" if len(flagged) else "#F0FDF4",
                "badge_fg": "#991B1B" if len(flagged) else "#166534",
            },
            {
                "label": "Anomaly Rate",
                "value": f"{anomaly_rate:.1%}",
                "subtitle": "Share of scanned sync records",
                "accent": "#2563EB",
            },
            {
                "label": "Critical Alerts",
                "value": f"{critical_count:,}",
                "subtitle": "Risk score >= 85 (priority review)",
                "accent": "#991B1B" if critical_count else "#16A34A",
                "badge": "Critical" if critical_count else "Clear",
                "badge_bg": "#FEF2F2" if critical_count else "#F0FDF4",
                "badge_fg": "#991B1B" if critical_count else "#166534",
            },
            {
                "label": "Overselling Exposure",
                "value": f"{oversell_units:,.0f} units",
                "subtitle": "Unexplained overstated channel stock",
                "accent": "#EA580C" if oversell_units > 0 else "#16A34A",
            },
        ]
    )

    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            show_chart(charts.category_bar(results))
    with c2:
        with st.container(border=True):
            show_chart(charts.severity_donut(results))

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        with st.container(border=True):
            show_chart(charts.channel_bar(results))
    with c4:
        with st.container(border=True):
            show_chart(charts.score_hist(results))

    tl_fig = charts.timeline(results)
    if tl_fig is not None:
        with st.container(border=True):
            show_chart(tl_fig)


elif page == "Detected Anomalies":
    if flagged.empty:
        with st.container(border=True):
            st.success("No anomalies detected at the current sensitivity setting. Try increasing sensitivity in the sidebar.")
    else:
        crit_n = int((flagged["severity"] == "Critical").sum())
        high_n = int((flagged["severity"] == "High").sum())
        med_n = int((flagged["severity"] == "Medium").sum())
        ui.render_kpi_row(
            [
                {
                    "label": "Total Flagged",
                    "value": f"{len(flagged):,}",
                    "subtitle": "Sorted highest risk first",
                    "accent": "#2563EB",
                },
                {
                    "label": "Critical Severity",
                    "value": f"{crit_n:,}",
                    "subtitle": "Risk score 85 - 100",
                    "accent": "#DC2626",
                    "badge": "Critical",
                    "badge_bg": "#FEF2F2",
                    "badge_fg": "#991B1B",
                },
                {
                    "label": "High Severity",
                    "value": f"{high_n:,}",
                    "subtitle": "Risk score 60 - 84",
                    "accent": "#EA580C",
                    "badge": "High",
                    "badge_bg": "#FFF7ED",
                    "badge_fg": "#9A3412",
                },
                {
                    "label": "Medium Severity",
                    "value": f"{med_n:,}",
                    "subtitle": "Risk score < 60",
                    "accent": "#D97706",
                    "badge": "Medium",
                    "badge_bg": "#FFFBEB",
                    "badge_fg": "#92400E",
                },
            ]
        )

        st.write("")
        with st.container(border=True):
            ui.render_section_header(
                "Filter & Search Anomalies",
                "Narrow flagged records by channel, issue category, severity tier, SKU, or minimum risk score.",
            )
            f1, f2, f3, f4, f5 = st.columns([1.1, 1.2, 1.0, 1.2, 1.0])
            sel_channels = f1.multiselect("Channel", sorted(flagged["channel"].unique()))
            sel_cats = f2.multiselect("Category", sorted(flagged["anomaly_category"].unique()))
            sel_sev = f3.multiselect("Severity", ["Critical", "High", "Medium"])
            sku_q = f4.text_input("Search SKU or reason", placeholder="e.g. SKU-100 or oversell")
            min_risk = f5.slider("Min risk score", 0, 100, 0)

            view = flagged
            if sel_channels:
                view = view[view["channel"].isin(sel_channels)]
            if sel_cats:
                view = view[view["anomaly_category"].isin(sel_cats)]
            if sel_sev:
                view = view[view["severity"].isin(sel_sev)]
            if sku_q:
                q_mask = (
                    view["sku"].str.contains(sku_q, case=False, na=False)
                    | view["explanation"].str.contains(sku_q, case=False, na=False)
                )
                view = view[q_mask]
            if min_risk > 0:
                view = view[view["risk_score"] >= min_risk]

            st.caption(f"Showing **{len(view):,}** of **{len(flagged):,}** flagged records (highest risk first).")

            cols = [
                "record_id",
                "timestamp",
                "sku",
                "channel",
                "severity",
                "risk_score",
                "anomaly_score",
                "anomaly_category",
                "explanation",
                "master_price",
                "channel_price",
                "master_stock",
                "channel_stock",
                "sync_delay_minutes",
            ]
            cols = [c for c in cols if c in view.columns]
            show_table(
                view[cols],
                column_config={
                    "record_id": st.column_config.TextColumn("Record ID"),
                    "sku": st.column_config.TextColumn("SKU"),
                    "channel": st.column_config.TextColumn("Channel"),
                    "severity": st.column_config.TextColumn("Severity"),
                    "risk_score": st.column_config.ProgressColumn("Risk (0-100)", min_value=0, max_value=100, format="%.0f"),
                    "anomaly_score": st.column_config.NumberColumn("Model Score", format="%.3f"),
                    "anomaly_category": st.column_config.TextColumn("Category"),
                    "explanation": st.column_config.TextColumn("Explanation", width="large"),
                    "master_price": st.column_config.NumberColumn("Master Price", format="$%.2f"),
                    "channel_price": st.column_config.NumberColumn("Channel Price", format="$%.2f"),
                    "master_stock": st.column_config.NumberColumn("Master Stock", format="%,.0f"),
                    "channel_stock": st.column_config.NumberColumn("Channel Stock", format="%,.0f"),
                    "sync_delay_minutes": st.column_config.NumberColumn("Delay (min)", format="%,.0f"),
                },
            )

            d1, d2, _ = st.columns([1.2, 1.2, 2.0])
            with d1:
                show_download_button(
                    "Download filtered anomalies (CSV)",
                    data=to_csv_bytes(view),
                    file_name="syncguard_filtered_anomalies.csv",
                    mime="text/csv",
                    key="dl_filtered",
                )
            with d2:
                show_download_button(
                    "Download all detected anomalies (CSV)",
                    data=to_csv_bytes(flagged),
                    file_name="syncguard_detected_anomalies.csv",
                    mime="text/csv",
                    key="dl_all",
                )

        with st.container(border=True):
            ui.render_section_header(
                "Record Inspector",
                "Select any flagged record to view its plain-English diagnosis and master-vs-channel comparison.",
            )
            if len(view):
                options = view["record_id"].head(500).tolist()
                rid = st.selectbox("Select record to inspect", options, label_visibility="collapsed")
                row = view[view["record_id"] == rid].iloc[0]
                ui.render_inspector_card(row)
            else:
                st.info("No records match the current filters. Adjust the filters above to inspect a record.")


elif page == "Pricing Analysis":
    price_flagged = int(flagged["anomaly_category"].isin(["Price Mismatch", "Multiple Issues"]).sum())
    max_dev_pct = float((results["price_diff_pct"].abs() * 100).max()) if len(results) else 0.0
    med_dev_pct = float((results["price_diff_pct"] * 100).median()) if len(results) else 0.0

    ui.render_kpi_row(
        [
            {
                "label": "Price-Related Alerts",
                "value": f"{price_flagged:,}",
                "subtitle": "Price mismatch or multi-issue records",
                "accent": "#DC2626" if price_flagged else "#16A34A",
            },
            {
                "label": "Median Price Diff",
                "value": f"{med_dev_pct:+.2f}%",
                "subtitle": "Channel price vs master catalogue",
                "accent": "#2563EB",
            },
            {
                "label": "Peak Price Deviation",
                "value": f"{max_dev_pct:,.1f}%",
                "subtitle": "Largest absolute discrepancy observed",
                "accent": "#EA580C",
            },
            {
                "label": "Learned Baselines",
                "value": f"{len(model.meta.get('learned_channel_markup_pct', {}))}",
                "subtitle": "Channels with calibrated normal markup",
                "accent": "#2563EB",
            },
        ]
    )

    st.write("")
    st.caption(
        "Every sales channel has its own normal markup learned from training data, so deviations are measured "
        "against each channel's expected baseline."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            show_chart(charts.price_hist(results))
    with c2:
        with st.container(border=True):
            show_chart(charts.price_box_by_channel(results))
    with st.container(border=True):
        show_chart(charts.price_scatter(results))


elif page == "Inventory Analysis":
    inv_flagged = int(flagged["anomaly_category"].isin(["Inventory Mismatch", "Multiple Issues"]).sum())
    oversell_units = float(flagged.loc[flagged["stock_residual"] > 0, "stock_residual"].sum()) if len(flagged) else 0.0
    false_stockout = int(((flagged["channel_stock"] <= 0) & (flagged["master_stock"] >= 5)).sum()) if len(flagged) else 0
    understated_n = int((flagged["stock_residual"] < 0).sum()) if len(flagged) else 0

    ui.render_kpi_row(
        [
            {
                "label": "Inventory Alerts",
                "value": f"{inv_flagged:,}",
                "subtitle": "Inventory mismatch or multi-issue",
                "accent": "#DC2626" if inv_flagged else "#16A34A",
            },
            {
                "label": "Overselling Exposure",
                "value": f"{oversell_units:,.0f} units",
                "subtitle": "Channel stock higher than master",
                "accent": "#EA580C",
            },
            {
                "label": "False Stock-Outs",
                "value": f"{false_stockout:,}",
                "subtitle": "Channel shows 0 while master >= 5",
                "accent": "#D97706",
            },
            {
                "label": "Understated Records",
                "value": f"{understated_n:,}",
                "subtitle": "Risking missed customer sales",
                "accent": "#2563EB",
            },
        ]
    )

    st.write("")
    st.caption(
        "A small gap between channel and master stock is normal due to recent sales since the last sync. "
        "The unexplained stock gap removes that expected sales lag."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            show_chart(charts.stock_gap_hist(results))
    with c2:
        with st.container(border=True):
            show_chart(charts.stock_direction_bar(results))
    with st.container(border=True):
        show_chart(charts.stock_scatter(results))


elif page == "Sync Delay Analysis":
    delay_flagged = int(flagged["anomaly_category"].isin(["Sync Delay", "Multiple Issues"]).sum())
    med_delay = float(results["sync_delay_minutes"].median()) if len(results) else 0.0
    max_delay = float(results["sync_delay_minutes"].max()) if len(results) else 0.0
    max_ratio = float(results["delay_ratio"].max()) if len(results) else 0.0

    ui.render_kpi_row(
        [
            {
                "label": "Sync Delay Alerts",
                "value": f"{delay_flagged:,}",
                "subtitle": "Sync delay or multi-issue records",
                "accent": "#DC2626" if delay_flagged else "#16A34A",
            },
            {
                "label": "Median Sync Delay",
                "value": f"{med_delay:,.0f} min",
                "subtitle": "Typical lag across loaded dataset",
                "accent": "#2563EB",
            },
            {
                "label": "Max Observed Delay",
                "value": f"{max_delay:,.0f} min",
                "subtitle": "Longest time since last sync",
                "accent": "#EA580C",
            },
            {
                "label": "Peak Cadence Multiple",
                "value": f"{max_ratio:,.1f}x",
                "subtitle": "Multiple of channel normal cadence",
                "accent": "#D97706",
            },
        ]
    )

    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            show_chart(charts.delay_hist(results))
    with c2:
        with st.container(border=True):
            show_chart(charts.delay_box_by_channel(results))
    with st.container(border=True):
        show_chart(charts.delay_vs_gap(results))


elif page == "Model Performance":
    if "is_anomaly" in results.columns:
        with st.container(border=True):
            ui.render_section_header(
                "Live Evaluation on Active Dataset",
                "Scored against the dataset's `is_anomaly` ground-truth column (never used during training). "
                "Adjust the sidebar Sensitivity slider to watch Precision and Recall shift live.",
            )
            y_true = results["is_anomaly"].fillna(0).astype(int)
            m = compute_metrics(y_true, results["predicted_anomaly"], results["anomaly_score"])
            ui.render_kpi_row(
                [
                    {
                        "label": "Precision",
                        "value": f"{m['precision']:.3f}",
                        "subtitle": f"TP {m['tp']:,} / Flagged {m['n_flagged']:,}",
                        "accent": "#2563EB",
                    },
                    {
                        "label": "Recall",
                        "value": f"{m['recall']:.3f}",
                        "subtitle": f"TP {m['tp']:,} / True {m['n_true_anomalies']:,}",
                        "accent": "#16A34A",
                    },
                    {
                        "label": "F1-Score",
                        "value": f"{m['f1']:.3f}",
                        "subtitle": "Harmonic mean of precision & recall",
                        "accent": "#2563EB",
                    },
                    {
                        "label": "ROC-AUC",
                        "value": f"{m.get('roc_auc', float('nan')):.3f}",
                        "subtitle": f"Accuracy: {m['accuracy']:.1%}",
                        "accent": "#0F172A",
                    },
                ]
            )
            st.write("")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                show_chart(charts.confusion_heatmap(m))
            with c2:
                if "anomaly_type" in results.columns:
                    rbt = recall_by_type(results)
                    if len(rbt):
                        show_chart(charts.recall_type_bar(rbt))
    else:
        st.info(
            "The currently loaded dataset has no `is_anomaly` label column, so live evaluation metrics are unavailable. "
            "Held-out training benchmark results are shown below."
        )

    saved = load_saved_metrics()
    with st.container(border=True):
        ui.render_section_header(
            "Saved Training Benchmark (Held-Out Test Set, Default Threshold)",
            "Recorded from `reports/evaluation_metrics.json` during pipeline evaluation.",
        )
        if saved:
            ui.render_kpi_row(
                [
                    {
                        "label": "Precision",
                        "value": f"{saved['precision']:.3f}",
                        "subtitle": f"TN {saved['tn']:,} | FP {saved['fp']:,}",
                        "accent": "#2563EB",
                    },
                    {
                        "label": "Recall",
                        "value": f"{saved['recall']:.3f}",
                        "subtitle": f"TP {saved['tp']:,} | FN {saved['fn']:,}",
                        "accent": "#16A34A",
                    },
                    {
                        "label": "F1-Score",
                        "value": f"{saved['f1']:.3f}",
                        "subtitle": f"ROC-AUC: {saved.get('roc_auc', float('nan')):.3f}",
                        "accent": "#2563EB",
                    },
                    {
                        "label": "PR-AUC",
                        "value": f"{saved.get('pr_auc', float('nan')):.3f}",
                        "subtitle": f"Test rows: {saved.get('n_records', 0):,}",
                        "accent": "#0F172A",
                    },
                ]
            )
            st.write("")
            sc1, sc2 = st.columns(2, gap="medium")
            with sc1:
                if saved.get("sensitivity_sweep"):
                    show_chart(charts.sensitivity_tradeoff_chart(saved["sensitivity_sweep"], float(sensitivity)))
            with sc2:
                if saved.get("recall_by_type"):
                    show_chart(charts.recall_type_bar(pd.DataFrame(saved["recall_by_type"])))

            with st.expander("View saved static evaluation charts (PNG)", expanded=False):
                imgs = [
                    "confusion_matrix.png",
                    "roc_pr_curves.png",
                    "score_distribution.png",
                    "recall_by_anomaly_type.png",
                    "sensitivity_tradeoff.png",
                    "pca_projection.png",
                ]
                cols = st.columns(2, gap="medium")
                for i, name in enumerate(imgs):
                    p = config.CHART_DIR / name
                    if p.exists():
                        cols[i % 2].image(str(p), caption=name.replace("_", " ").replace(".png", "").title())
        else:
            st.info("No saved evaluation found. Run `python run_pipeline.py`.")

    with st.container(border=True):
        ui.render_section_header(
            "What the Model Learned",
            "Unsupervised per-channel markups, normal operating tolerances, and engineered features.",
        )
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            st.markdown("**Normal price markup per channel (%)**")
            markups = model.meta.get("learned_channel_markup_pct", {})
            if markups:
                show_table(
                    pd.DataFrame(
                        [{"Channel": k, "Learned Markup (%)": f"{v:+.2f}%"} for k, v in markups.items()]
                    )
                )
        with m2:
            st.markdown("**Explanation tolerances (from normal records)**")
            tols = model.meta.get("tolerances", {})
            if tols:
                show_table(
                    pd.DataFrame(
                        [{"Tolerance Parameter": k.replace("_", " ").title(), "Value": str(v)} for k, v in tols.items()]
                    )
                )
        st.markdown("**Model input features**")
        st.code(", ".join(model.meta.get("features", [])))


elif page == "Upload & Schema":
    c_up, c_req = st.columns([1.1, 1.0], gap="medium")
    with c_up:
        with st.container(border=True):
            ui.render_section_header(
                "Upload Custom Sync CSV",
                "Upload a CSV file here or from the sidebar. Make sure 'Upload my own CSV' is selected in Data Source to scan it.",
            )
            page_upload = st.file_uploader(
                "Select or drop a CSV file",
                type=["csv"],
                key="page_uploader",
            )
            if page_upload is not None:
                _store_uploaded_file(page_upload)
                if not source.startswith("Upload"):
                    st.success(
                        f"Loaded **{page_upload.name}**. Switch **Data Source** in the sidebar to "
                        "`Upload my own CSV` to run detection on it."
                    )
            if config.SAMPLE_PATH.exists():
                st.download_button(
                    "Download sample CSV template (600 rows)",
                    data=config.SAMPLE_PATH.read_bytes(),
                    file_name="syncguard_sample.csv",
                    mime="text/csv",
                    key="page_dl_sample",
                )

    with c_req:
        with st.container(border=True):
            ui.render_section_header(
                "Active Dataset Column Check",
                "Green pills indicate columns detected in the currently loaded dataset.",
            )
            active_cols = set(normalise_columns(df_raw).columns)
            ui.render_validation_checklist(active_cols, config.REQUIRED_COLUMNS, config.OPTIONAL_COLUMNS)

    with st.container(border=True):
        ui.render_section_header("CSV Format Specification", "Column names are case-insensitive; spaces become underscores.")
        show_table(_schema_dataframe())

    with st.container(border=True):
        ui.render_section_header("Active Dataset Preview", "Showing the first 200 processed records.")
        show_table(results.drop(columns=["explanation"]).head(200))
