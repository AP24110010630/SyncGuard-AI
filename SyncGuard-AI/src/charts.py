"""Plotly figures used by the Streamlit dashboard.

Every function takes the *results* dataframe produced by
``SyncGuardModel.predict`` and returns a ``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NORMAL = "#2563eb"
ANOMALY = "#dc2626"
SEVERITY_COLORS = {"Critical": "#7f1d1d", "High": "#dc2626", "Medium": "#f59e0b"}


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        **kw,
    )
    return fig


def _split(res: pd.DataFrame):
    return res[res["predicted_anomaly"] == 0], res[res["predicted_anomaly"] == 1]


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data to display", showarrow=False, font=dict(size=14, color="#64748b"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, 300)


# ------------------------------------------------------------------ overview
def category_bar(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Anomalies by category")
    c = an["anomaly_category"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=ANOMALY, text=c.values, textposition="outside"))
    fig.update_xaxes(title="Flagged records")
    return _layout(fig, "Anomalies by category")


def channel_bar(res: pd.DataFrame) -> go.Figure:
    g = res.groupby("channel")["predicted_anomaly"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"] * 100
    g = g.sort_values("rate", ascending=False)
    fig = go.Figure(
        go.Bar(x=g.index, y=g["rate"], marker_color=ANOMALY,
               text=[f"{int(s)} / {int(n)}" for s, n in zip(g["sum"], g["count"])], textposition="outside")
    )
    fig.update_yaxes(title="Anomaly rate (%)")
    return _layout(fig, "Anomaly rate by channel (flagged / total)")


def score_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_histogram(x=nm["anomaly_score"], name="Normal", marker_color=NORMAL, opacity=0.75, nbinsx=60)
    fig.add_histogram(x=an["anomaly_score"], name="Flagged", marker_color=ANOMALY, opacity=0.8, nbinsx=60)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Anomaly score (higher = more anomalous)")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Anomaly score distribution")


def timeline(res: pd.DataFrame):
    if "timestamp" not in res.columns or res["timestamp"].isna().all():
        return None
    t = res.dropna(subset=["timestamp"]).copy()
    span_days = (t["timestamp"].max() - t["timestamp"].min()).total_seconds() / 86400
    freq = "h" if span_days <= 3 else "D"
    t["bucket"] = t["timestamp"].dt.floor(freq)
    g = t.groupby("bucket")["predicted_anomaly"].sum().reset_index()
    fig = go.Figure(go.Bar(x=g["bucket"], y=g["predicted_anomaly"], marker_color=ANOMALY))
    fig.update_yaxes(title="Flagged records")
    return _layout(fig, f"Flagged anomalies over time ({'hourly' if freq == 'h' else 'daily'})")


def severity_donut(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Severity mix")
    c = an["severity"].value_counts().reindex(["Critical", "High", "Medium"]).dropna()
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.55,
                           marker=dict(colors=[SEVERITY_COLORS[k] for k in c.index])))
    return _layout(fig, "Severity mix")


# --------------------------------------------------------------------- price
def price_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col in ((nm, "Normal", NORMAL), (an, "Flagged", ANOMALY)):
        fig.add_histogram(x=(d["price_diff_pct"] * 100).clip(-50, 50), name=name, marker_color=col, opacity=0.75, nbinsx=80)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Channel price vs master price (%) - clipped to +/-50%")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Pricing differences")


def price_box_by_channel(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for flag, name, col in ((0, "Normal", NORMAL), (1, "Flagged", ANOMALY)):
        d = res[res["predicted_anomaly"] == flag]
        fig.add_trace(go.Box(x=d["channel"], y=(d["price_diff_pct"] * 100).clip(-50, 50), name=name, marker_color=col, boxpoints="outliers"))
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title="Price difference (%)")
    return _layout(fig, "Price difference by channel (each channel has its own normal markup)")


def price_scatter(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=nm["master_price"], y=nm["channel_price"], mode="markers", name="Normal",
                               marker=dict(color=NORMAL, size=4, opacity=0.4)))
    fig.add_trace(go.Scattergl(x=an["master_price"], y=an["channel_price"], mode="markers", name="Flagged",
                               marker=dict(color=ANOMALY, size=7, opacity=0.85),
                               text=an["sku"] + " | " + an["channel"], hoverinfo="text+x+y"))
    pos = res.loc[(res["master_price"] > 0) & (res["channel_price"] > 0)]
    if len(pos):
        lo, hi = float(pos["master_price"].min()), float(pos["master_price"].max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect sync",
                                 line=dict(color="#64748b", dash="dash")))
    fig.update_xaxes(title="Master price", type="log")
    fig.update_yaxes(title="Channel price", type="log")
    return _layout(fig, "Master vs channel price (log scale)", 420)


# ----------------------------------------------------------------- inventory
def stock_scatter(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=nm["master_stock"], y=nm["channel_stock"], mode="markers", name="Normal",
                               marker=dict(color=NORMAL, size=4, opacity=0.4)))
    fig.add_trace(go.Scattergl(x=an["master_stock"], y=an["channel_stock"], mode="markers", name="Flagged",
                               marker=dict(color=ANOMALY, size=7, opacity=0.85),
                               text=an["sku"] + " | " + an["channel"], hoverinfo="text+x+y"))
    hi = float(max(res["master_stock"].max(), res["channel_stock"].max(), 1))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="Perfect sync",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_xaxes(title="Master stock (units)")
    fig.update_yaxes(title="Channel stock (units)")
    return _layout(fig, "Inventory mismatch: master vs channel stock", 420)


def stock_gap_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col in ((nm, "Normal", NORMAL), (an, "Flagged", ANOMALY)):
        fig.add_histogram(x=d["stock_residual"].clip(-60, 120), name=name, marker_color=col, opacity=0.75, nbinsx=80)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Unexplained stock gap (channel - master - expected sales lag), clipped")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Unexplained stock gap distribution")


def stock_direction_bar(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Flagged stock gaps by direction")
    a = an.copy()
    a["direction"] = np.where(a["stock_residual"] > 0, "Overstated (overselling risk)", "Understated (lost sales)")
    g = a.groupby(["channel", "direction"]).size().unstack(fill_value=0)
    fig = go.Figure()
    colors = {"Overstated (overselling risk)": ANOMALY, "Understated (lost sales)": "#f59e0b"}
    for col in g.columns:
        fig.add_bar(x=g.index, y=g[col], name=col, marker_color=colors.get(col, NORMAL))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Flagged records")
    return _layout(fig, "Flagged records by stock-gap direction")


# --------------------------------------------------------------- sync delay
def delay_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col in ((nm, "Normal", NORMAL), (an, "Flagged", ANOMALY)):
        fig.add_histogram(x=d["delay_ratio"].clip(0, 20), name=name, marker_color=col, opacity=0.75, nbinsx=80)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Sync delay as a multiple of the channel's normal cadence (clipped at 20x)")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Synchronisation delays")


def delay_box_by_channel(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for flag, name, col in ((0, "Normal", NORMAL), (1, "Flagged", ANOMALY)):
        d = res[res["predicted_anomaly"] == flag]
        fig.add_trace(go.Box(x=d["channel"], y=d["sync_delay_minutes"], name=name, marker_color=col, boxpoints="outliers"))
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title="Sync delay (minutes)", type="log")
    return _layout(fig, "Sync delay by channel (log scale)")


def delay_vs_gap(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=nm["delay_ratio"].clip(0, 40), y=nm["stock_residual"].abs().clip(0, 150), mode="markers",
                               name="Normal", marker=dict(color=NORMAL, size=4, opacity=0.35)))
    fig.add_trace(go.Scattergl(x=an["delay_ratio"].clip(0, 40), y=an["stock_residual"].abs().clip(0, 150), mode="markers",
                               name="Flagged", marker=dict(color=ANOMALY, size=7, opacity=0.85),
                               text=an["sku"] + " | " + an["channel"], hoverinfo="text+x+y"))
    fig.update_xaxes(title="Sync delay (x cadence, clipped)")
    fig.update_yaxes(title="|Unexplained stock gap| (units, clipped)")
    return _layout(fig, "Do long delays coincide with stock gaps?", 420)


# --------------------------------------------------------------- evaluation
def confusion_heatmap(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted anomaly"], y=["Actual normal", "Actual anomaly"],
                               colorscale="Blues", showscale=False, text=[[f"{v:,}" for v in r] for r in z],
                               texttemplate="%{text}", textfont=dict(size=22)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Confusion matrix", 360)


def recall_type_bar(rbt: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=rbt["recall"] * 100, y=rbt["anomaly_type"], orientation="h", marker_color=NORMAL,
                           text=[f"{r:.0f}%" for r in rbt["recall"] * 100], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Recall (%)", range=[0, 110])
    return _layout(fig, "Detection rate by anomaly type", 340)
