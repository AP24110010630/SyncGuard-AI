"""Plotly figures used by the Streamlit dashboard.

Every function takes the *results* dataframe produced by
``SyncGuardModel.predict`` (or evaluation dicts) and returns a
``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NORMAL = "#2563EB"
ANOMALY = "#DC2626"
AMBER = "#F59E0B"
GREEN = "#16A34A"
SLATE_TEXT = "#475569"
NAVY_TEXT = "#0F172A"
GRID_COLOR = "#F1F5F9"
ZERO_COLOR = "#E2E8F0"

SEVERITY_COLORS = {
    "Critical": "#DC2626",
    "High": "#EA580C",
    "Medium": "#F59E0B",
}


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            y=0.96,
            font=dict(size=15, color=NAVY_TEXT, family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            color=SLATE_TEXT,
            size=12,
        ),
        height=height,
        margin=dict(l=44, r=22, t=58, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11, color=SLATE_TEXT),
        ),
        hoverlabel=dict(
            bgcolor=NAVY_TEXT,
            bordercolor=NAVY_TEXT,
            font=dict(color="#FFFFFF", size=12),
        ),
        **kw,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_COLOR,
        title_font=dict(size=12, color=SLATE_TEXT),
        tickfont=dict(size=11, color=SLATE_TEXT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_COLOR,
        title_font=dict(size=12, color=SLATE_TEXT),
        tickfont=dict(size=11, color=SLATE_TEXT),
    )
    return fig


def _split(res: pd.DataFrame):
    return res[res["predicted_anomaly"] == 0], res[res["predicted_anomaly"] == 1]


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No anomalies to display at this setting",
        showarrow=False,
        font=dict(size=13, color="#64748B"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, 320)


# ------------------------------------------------------------------ overview
def category_bar(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Anomalies by category")
    c = an["anomaly_category"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=c.values,
            y=c.index,
            orientation="h",
            marker=dict(color=ANOMALY, opacity=0.88),
            text=[f"{int(v):,}" for v in c.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Flagged: %{x:,}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Flagged records", rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, "Anomalies by category", 370)


def channel_bar(res: pd.DataFrame) -> go.Figure:
    g = res.groupby("channel")["predicted_anomaly"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"] * 100
    g = g.sort_values("rate", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=g.index,
            y=g["rate"],
            marker=dict(color=NORMAL, opacity=0.9),
            text=[f"{r:.1f}% ({int(s)}/{int(n)})" for r, s, n in zip(g["rate"], g["sum"], g["count"])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Anomaly rate: %{y:.2f}%<br>%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="Anomaly rate (%)", rangemode="tozero")
    return _layout(fig, "Anomaly rate by channel (flagged / total)", 370)


def score_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_histogram(
        x=nm["anomaly_score"],
        name="Normal",
        marker_color=NORMAL,
        opacity=0.65,
        nbinsx=60,
        hovertemplate="Score: %{x:.3f}<br>Normal records: %{y:,}<extra></extra>",
    )
    fig.add_histogram(
        x=an["anomaly_score"],
        name="Flagged",
        marker_color=ANOMALY,
        opacity=0.82,
        nbinsx=60,
        hovertemplate="Score: %{x:.3f}<br>Flagged records: %{y:,}<extra></extra>",
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Anomaly score (higher = more anomalous)")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Anomaly score distribution", 370)


def timeline(res: pd.DataFrame):
    if "timestamp" not in res.columns or res["timestamp"].isna().all():
        return None
    t = res.dropna(subset=["timestamp"]).copy()
    span_days = (t["timestamp"].max() - t["timestamp"].min()).total_seconds() / 86400
    freq = "h" if span_days <= 3 else "D"
    t["bucket"] = t["timestamp"].dt.floor(freq)
    g = t.groupby("bucket")["predicted_anomaly"].agg(["sum", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=g["bucket"],
            y=g["sum"],
            name="Flagged anomalies",
            marker=dict(color=ANOMALY, opacity=0.82),
            customdata=g["count"],
            hovertemplate="<b>%{x}</b><br>Flagged: %{y:,}<br>Total scanned: %{customdata:,}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Flagged records", rangemode="tozero")
    return _layout(fig, f"Flagged anomalies over time ({'hourly' if freq == 'h' else 'daily'})", 340)


def severity_donut(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Severity mix")
    c = an["severity"].value_counts().reindex(["Critical", "High", "Medium"]).dropna()
    fig = go.Figure(
        go.Pie(
            labels=c.index,
            values=c.values,
            hole=0.62,
            marker=dict(
                colors=[SEVERITY_COLORS[k] for k in c.index],
                line=dict(color="#FFFFFF", width=2),
            ),
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{len(an):,}</b><br><span style='font-size:11px;color:#64748B'>Flagged</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, color=NAVY_TEXT),
    )
    return _layout(fig, "Severity mix", 370)


# --------------------------------------------------------------------- price
def price_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col, op in ((nm, "Normal", NORMAL, 0.65), (an, "Flagged", ANOMALY, 0.82)):
        fig.add_histogram(
            x=(d["price_diff_pct"] * 100).clip(-50, 50),
            name=name,
            marker_color=col,
            opacity=op,
            nbinsx=80,
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Channel price vs master price (%) - clipped to +/-50%")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Pricing differences distribution", 370)


def price_box_by_channel(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for flag, name, col in ((0, "Normal", NORMAL), (1, "Flagged", ANOMALY)):
        d = res[res["predicted_anomaly"] == flag]
        fig.add_trace(
            go.Box(
                x=d["channel"],
                y=(d["price_diff_pct"] * 100).clip(-50, 50),
                name=name,
                marker_color=col,
                boxpoints="outliers",
            )
        )
    fig.update_layout(boxmode="group")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="Price difference (%)")
    return _layout(fig, "Price difference by channel (channel baseline markup)", 370)


def price_scatter(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=nm["master_price"],
            y=nm["channel_price"],
            mode="markers",
            name="Normal",
            marker=dict(color=NORMAL, size=4, opacity=0.35),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=an["master_price"],
            y=an["channel_price"],
            mode="markers",
            name="Flagged",
            marker=dict(color=ANOMALY, size=7, opacity=0.85, line=dict(color="#FFFFFF", width=0.5)),
            customdata=np.stack([an["sku"], an["channel"], an["severity"]], axis=-1) if len(an) else None,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Severity: %{customdata[2]}<br>"
                "Master: $%{x:,.2f}<br>Channel: $%{y:,.2f}<extra></extra>"
            ),
        )
    )
    pos = res.loc[(res["master_price"] > 0) & (res["channel_price"] > 0)]
    if len(pos):
        lo, hi = float(pos["master_price"].min()), float(pos["master_price"].max())
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect sync",
                line=dict(color="#64748B", dash="dash", width=1.5),
            )
        )
    fig.update_xaxes(title="Master price", type="log")
    fig.update_yaxes(title="Channel price", type="log")
    return _layout(fig, "Master vs channel price (log scale)", 430)


# ----------------------------------------------------------------- inventory
def stock_scatter(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=nm["master_stock"],
            y=nm["channel_stock"],
            mode="markers",
            name="Normal",
            marker=dict(color=NORMAL, size=4, opacity=0.35),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=an["master_stock"],
            y=an["channel_stock"],
            mode="markers",
            name="Flagged",
            marker=dict(color=ANOMALY, size=7, opacity=0.85, line=dict(color="#FFFFFF", width=0.5)),
            customdata=np.stack([an["sku"], an["channel"], an["severity"]], axis=-1) if len(an) else None,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Severity: %{customdata[2]}<br>"
                "Master stock: %{x:,.0f}<br>Channel stock: %{y:,.0f}<extra></extra>"
            ),
        )
    )
    hi = float(max(res["master_stock"].max(), res["channel_stock"].max(), 1))
    fig.add_trace(
        go.Scatter(
            x=[0, hi],
            y=[0, hi],
            mode="lines",
            name="Perfect sync",
            line=dict(color="#64748B", dash="dash", width=1.5),
        )
    )
    fig.update_xaxes(title="Master stock (units)")
    fig.update_yaxes(title="Channel stock (units)")
    return _layout(fig, "Inventory mismatch: master vs channel stock", 430)


def stock_gap_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col, op in ((nm, "Normal", NORMAL, 0.65), (an, "Flagged", ANOMALY, 0.82)):
        fig.add_histogram(
            x=d["stock_residual"].clip(-60, 120),
            name=name,
            marker_color=col,
            opacity=op,
            nbinsx=80,
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Unexplained stock gap (channel - master - expected sales lag), clipped")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Unexplained stock gap distribution", 370)


def stock_direction_bar(res: pd.DataFrame) -> go.Figure:
    _, an = _split(res)
    if an.empty:
        return _empty("Flagged stock gaps by direction")
    a = an.copy()
    a["direction"] = np.where(a["stock_residual"] > 0, "Overstated (overselling risk)", "Understated (lost sales)")
    g = a.groupby(["channel", "direction"]).size().unstack(fill_value=0)
    fig = go.Figure()
    colors = {"Overstated (overselling risk)": ANOMALY, "Understated (lost sales)": AMBER}
    for col in g.columns:
        fig.add_bar(
            x=g.index,
            y=g[col],
            name=col,
            marker_color=colors.get(col, NORMAL),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="Flagged records")
    return _layout(fig, "Flagged records by stock-gap direction", 370)


# --------------------------------------------------------------- sync delay
def delay_hist(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    for d, name, col, op in ((nm, "Normal", NORMAL, 0.65), (an, "Flagged", ANOMALY, 0.82)):
        fig.add_histogram(
            x=d["delay_ratio"].clip(0, 20),
            name=name,
            marker_color=col,
            opacity=op,
            nbinsx=80,
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Sync delay as a multiple of the channel's normal cadence (clipped at 20x)")
    fig.update_yaxes(title="Records")
    return _layout(fig, "Synchronisation delays distribution", 370)


def delay_box_by_channel(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for flag, name, col in ((0, "Normal", NORMAL), (1, "Flagged", ANOMALY)):
        d = res[res["predicted_anomaly"] == flag]
        fig.add_trace(
            go.Box(
                x=d["channel"],
                y=d["sync_delay_minutes"],
                name=name,
                marker_color=col,
                boxpoints="outliers",
            )
        )
    fig.update_layout(boxmode="group")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="Sync delay (minutes)", type="log")
    return _layout(fig, "Sync delay by channel (log scale)", 370)


def delay_vs_gap(res: pd.DataFrame) -> go.Figure:
    nm, an = _split(res)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=nm["delay_ratio"].clip(0, 40),
            y=nm["stock_residual"].abs().clip(0, 150),
            mode="markers",
            name="Normal",
            marker=dict(color=NORMAL, size=4, opacity=0.3),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=an["delay_ratio"].clip(0, 40),
            y=an["stock_residual"].abs().clip(0, 150),
            mode="markers",
            name="Flagged",
            marker=dict(color=ANOMALY, size=7, opacity=0.85, line=dict(color="#FFFFFF", width=0.5)),
            customdata=np.stack([an["sku"], an["channel"], an["severity"]], axis=-1) if len(an) else None,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Severity: %{customdata[2]}<br>"
                "Delay ratio: %{x:.1f}x<br>|Unexplained gap|: %{y:.1f} units<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Sync delay (x cadence, clipped)")
    fig.update_yaxes(title="|Unexplained stock gap| (units, clipped)")
    return _layout(fig, "Do long delays coincide with stock gaps?", 430)


# --------------------------------------------------------------- evaluation
def confusion_heatmap(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted normal", "Predicted anomaly"],
            y=["Actual normal", "Actual anomaly"],
            colorscale=[[0, "#EFF6FF"], [0.4, "#93C5FD"], [1, "#1D4ED8"]],
            showscale=False,
            text=[[f"{v:,}" for v in r] for r in z],
            texttemplate="%{text}",
            textfont=dict(size=20),
            hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, "Confusion matrix", 360)


def recall_type_bar(rbt: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=rbt["recall"] * 100,
            y=rbt["anomaly_type"],
            orientation="h",
            marker=dict(color=NORMAL, opacity=0.9),
            text=[f"{r * 100:.0f}% ({int(d)}/{int(t)})" for r, d, t in zip(rbt["recall"], rbt["detected"], rbt["total"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Recall: %{x:.1f}%<br>%{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Recall (%)", range=[0, 118])
    return _layout(fig, "Detection rate by anomaly type", 360)


def sensitivity_tradeoff_chart(sweep: dict, current_sensitivity: float = 50.0) -> go.Figure:
    """Interactive Precision / Recall / F1 trade-off from saved evaluation sweep."""
    xs = sweep.get("sensitivity", [])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=sweep.get("precision", []),
            mode="lines",
            name="Precision",
            line=dict(color=NORMAL, width=2.5),
            hovertemplate="Sensitivity %{x:.1f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=sweep.get("recall", []),
            mode="lines",
            name="Recall",
            line=dict(color=ANOMALY, width=2.5),
            hovertemplate="Sensitivity %{x:.1f}<br>Recall: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=sweep.get("f1", []),
            mode="lines",
            name="F1-score",
            line=dict(color=GREEN, width=2.5, dash="dash"),
            hovertemplate="Sensitivity %{x:.1f}<br>F1-score: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=float(current_sensitivity),
        line_dash="dot",
        line_color="#64748B",
        annotation_text=f"Current ({int(current_sensitivity)})",
        annotation_position="top right",
    )
    fig.update_xaxes(title="Sensitivity slider (0-100)")
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _layout(fig, "Precision / Recall / F1 trade-off across sensitivity", 370)
