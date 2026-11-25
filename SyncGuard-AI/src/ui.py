"""Reusable SaaS UI styling and HTML components for SyncGuard AI.

All presentation helpers live here so ``app.py`` stays focused on dashboard
flow and data binding while preserving 100% of the underlying ML logic.
"""
from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ Palette
NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#64748B"
BORDER = "#E2E8F0"
CANVAS = "#F8FAFC"
SURFACE = "#FFFFFF"
BLUE = "#2563EB"
BLUE_SOFT = "#EFF6FF"
BLUE_DARK = "#1D4ED8"

SEVERITY_STYLES = {
    "Critical": {
        "bg": "#FEF2F2",
        "border": "#FECACA",
        "text": "#991B1B",
        "dot": "#DC2626",
    },
    "High": {
        "bg": "#FFF7ED",
        "border": "#FED7AA",
        "text": "#9A3412",
        "dot": "#EA580C",
    },
    "Medium": {
        "bg": "#FFFBEB",
        "border": "#FDE68A",
        "text": "#92400E",
        "dot": "#D97706",
    },
    "Normal": {
        "bg": "#F0FDF4",
        "border": "#BBF7D0",
        "text": "#166534",
        "dot": "#16A34A",
    },
}


def _e(value: object) -> str:
    text = " ".join(str(value).splitlines())
    return html.escape(text)


def _render_html(target, markup: str) -> None:
    """Strip indentation and blank lines so CommonMark renders HTML cleanly."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    target.markdown(cleaned, unsafe_allow_html=True)


def inject_global_styles() -> None:
    """Inject consistent light-theme SaaS typography, card, and sidebar styles."""
    _render_html(
        st,
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {NAVY};
        }}

        .stApp {{
            background-color: {CANVAS};
        }}

        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1440px;
        }}

        /* Headings */
        h1, h2, h3, h4 {{
            color: {NAVY};
            letter-spacing: -0.02em;
        }}

        /* Sidebar shell */
        section[data-testid="stSidebar"] {{
            background-color: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.25rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }}

        /* Sidebar brand block */
        .sg-side-brand {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.35rem 0.25rem 0.9rem 0.25rem;
            border-bottom: 1px solid {BORDER};
            margin-bottom: 0.9rem;
        }}
        .sg-side-logo {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1rem;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22);
            flex-shrink: 0;
        }}
        .sg-side-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.15;
        }}
        .sg-side-sub {{
            font-size: 0.75rem;
            color: {MUTED};
            margin-top: 2px;
        }}
        .sg-side-label {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {MUTED};
            margin: 0.95rem 0 0.45rem 0;
        }}

        /* Sidebar radio items styled as clean SaaS rows */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 0.28rem;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 9px;
            padding: 0.42rem 0.65rem;
            margin: 0;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: #F1F5F9;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            background-color: {BLUE_SOFT};
            border-color: #BFDBFE;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
            color: {BLUE_DARK};
            font-weight: 600;
        }}

        /* Model status box in sidebar */
        .sg-model-card {{
            background: {CANVAS};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
            margin-top: 0.75rem;
        }}
        .sg-model-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #166534;
            margin-bottom: 0.35rem;
        }}
        .sg-dot-green {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #16A34A;
            display: inline-block;
        }}
        .sg-model-meta {{
            font-size: 0.76rem;
            color: {SLATE};
            line-height: 1.45;
        }}

        /* Top workspace header */
        .sg-topbar {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.15rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }}
        .sg-topbar-kicker {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {BLUE};
            margin-bottom: 0.2rem;
        }}
        .sg-topbar-title {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.2;
            margin: 0;
        }}
        .sg-topbar-sub {{
            font-size: 0.88rem;
            color: {MUTED};
            margin-top: 0.25rem;
        }}
        .sg-pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }}
        .sg-pill {{
            background: {CANVAS};
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 0.32rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 500;
            color: {SLATE};
            white-space: nowrap;
        }}
        .sg-pill strong {{
            color: {NAVY};
            font-weight: 600;
        }}
        .sg-pill-blue {{
            background: {BLUE_SOFT};
            border-color: #BFDBFE;
            color: {BLUE_DARK};
        }}

        /* KPI cards */
        .sg-kpi-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-top: 3px solid var(--accent, {BLUE});
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            height: 100%;
        }}
        .sg-kpi-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
        }}
        .sg-kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .sg-kpi-badge {{
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            background: var(--badge-bg, {BLUE_SOFT});
            color: var(--badge-fg, {BLUE_DARK});
        }}
        .sg-kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.15;
            letter-spacing: -0.02em;
        }}
        .sg-kpi-sub {{
            font-size: 0.78rem;
            color: {MUTED};
            margin-top: 0.32rem;
        }}

        /* Card containers (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }}

        /* Section headers inside pages */
        .sg-sec-head {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
            margin: 0.35rem 0 0.75rem 0;
        }}
        .sg-sec-title {{
            font-size: 1.08rem;
            font-weight: 700;
            color: {NAVY};
            margin: 0;
        }}
        .sg-sec-sub {{
            font-size: 0.84rem;
            color: {MUTED};
            margin: 0.15rem 0 0 0;
        }}

        /* Severity pill badge */
        .sg-sev-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
            white-space: nowrap;
        }}
        .sg-sev-dot {{
            width: 7px;
            height: 7px;
            border-radius: 999px;
            display: inline-block;
        }}

        /* Inspector card */
        .sg-inspector-banner {{
            background: {CANVAS};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin: 0.65rem 0 0.9rem 0;
        }}
        .sg-inspector-row {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.6rem;
        }}
        .sg-inspector-exp {{
            font-size: 0.92rem;
            color: {NAVY};
            line-height: 1.55;
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 9px;
            padding: 0.75rem 0.95rem;
        }}
        .sg-cmp-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 11px;
            padding: 0.85rem 1rem;
        }}
        .sg-cmp-title {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {MUTED};
            margin-bottom: 0.35rem;
        }}
        .sg-cmp-val {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {NAVY};
        }}
        .sg-cmp-sub {{
            font-size: 0.8rem;
            color: {SLATE};
            margin-top: 0.25rem;
        }}

        /* Upload dropzone & schema checklist */
        section[data-testid="stFileUploaderDropzone"] {{
            background: {BLUE_SOFT};
            border: 1.5px dashed #93C5FD;
            border-radius: 12px;
            padding: 1rem;
        }}
        .sg-check-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.45rem;
        }}
        .sg-check-ok {{
            background: #F0FDF4;
            color: #166534;
            border: 1px solid #BBF7D0;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.77rem;
            font-weight: 600;
        }}
        .sg-check-miss {{
            background: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.77rem;
            font-weight: 600;
        }}
        .sg-check-opt {{
            background: {CANVAS};
            color: {MUTED};
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.77rem;
            font-weight: 500;
        }}

        /* Buttons & download buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 9px;
            font-weight: 600;
            border: 1px solid {BORDER};
            transition: all 0.15s ease;
        }}
        .stDownloadButton > button:hover, .stButton > button:hover {{
            border-color: {BLUE};
            color: {BLUE};
        }}

        /* Default Streamlit metric fallback */
        div[data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            padding: 0.85rem 1rem;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }}
        </style>
        """,
    )


def render_sidebar_brand() -> None:
    """Render the SyncGuard AI logo and brand subtitle at the top of the sidebar."""
    _render_html(
        st,
        """
        <div class="sg-side-brand">
            <div class="sg-side-logo">SG</div>
            <div>
                <div class="sg-side-title">SyncGuard AI</div>
                <div class="sg-side-sub">Multi-Channel Sync Monitor</div>
            </div>
        </div>
        """,
    )


def render_sidebar_section_label(text: str) -> None:
    _render_html(st, f'<div class="sg-side-label">{_e(text)}</div>')


def render_sidebar_model_card(n_train_rows: object, channels: Iterable[str]) -> None:
    """Render a compact model health summary card in the sidebar."""
    rows_txt = f"{n_train_rows:,}" if isinstance(n_train_rows, (int, float)) else str(n_train_rows)
    chan_count = len(list(channels)) if channels else 0
    _render_html(
        st,
        f"""
        <div class="sg-model-card">
            <div class="sg-model-status">
                <span class="sg-dot-green"></span>
                Isolation Forest Ready
            </div>
            <div class="sg-model-meta">
                Unsupervised baseline trained on <strong>{_e(rows_txt)}</strong> records
                across <strong>{_e(chan_count)}</strong> learned channels.
            </div>
        </div>
        """,
    )


def render_top_bar(
    page_title: str,
    page_subtitle: str,
    dataset_label: str,
    n_records: int | None = None,
    n_flagged: int | None = None,
    sensitivity: int = 50,
) -> None:
    """Render the top workspace banner with live context pills."""
    pills = [f'<span class="sg-pill">Source: <strong>{_e(dataset_label)}</strong></span>']
    if n_records is not None and n_flagged is not None:
        rate = n_flagged / max(n_records, 1)
        pills.append(f'<span class="sg-pill">Scanned: <strong>{n_records:,}</strong></span>')
        pills.append(
            f'<span class="sg-pill sg-pill-blue">Flagged: <strong>{n_flagged:,}</strong> ({rate:.1%})</span>'
        )
    pills.append(f'<span class="sg-pill">Sensitivity: <strong>{int(sensitivity)}/100</strong></span>')

    _render_html(
        st,
        f"""
        <div class="sg-topbar">
            <div>
                <div class="sg-topbar-kicker">E-Commerce Sync Intelligence</div>
                <div class="sg-topbar-title">{_e(page_title)}</div>
                <div class="sg-topbar-sub">{_e(page_subtitle)}</div>
            </div>
            <div class="sg-pill-row">
                {''.join(pills)}
            </div>
        </div>
        """,
    )


def render_kpi_row(cards: list[dict]) -> None:
    """Render a row of styled SaaS KPI cards.

    Each card dict supports: ``label``, ``value``, ``subtitle``, ``accent``,
    and optional ``badge``, ``badge_bg``, ``badge_fg``.
    """
    cols = st.columns(len(cards), gap="medium")
    for col, card in zip(cols, cards):
        accent = card.get("accent", BLUE)
        badge_html = ""
        if card.get("badge"):
            bg = card.get("badge_bg", BLUE_SOFT)
            fg = card.get("badge_fg", BLUE_DARK)
            badge_html = (
                f'<span class="sg-kpi-badge" style="--badge-bg:{bg};--badge-fg:{fg};">'
                f'{_e(card["badge"])}</span>'
            )
        _render_html(
            col,
            f"""
            <div class="sg-kpi-card" style="--accent:{accent};">
                <div class="sg-kpi-top">
                    <span class="sg-kpi-label">{_e(card["label"])}</span>
                    {badge_html}
                </div>
                <div class="sg-kpi-value">{_e(card["value"])}</div>
                <div class="sg-kpi-sub">{_e(card.get("subtitle", ""))}</div>
            </div>
            """,
        )


def render_section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<p class="sg-sec-sub">{_e(subtitle)}</p>' if subtitle else ""
    _render_html(
        st,
        f"""
        <div class="sg-sec-head">
            <div>
                <h3 class="sg-sec-title">{_e(title)}</h3>
                {sub_html}
            </div>
        </div>
        """,
    )


def severity_badge_html(severity: str) -> str:
    label = severity if severity in SEVERITY_STYLES else "Normal"
    st_cfg = SEVERITY_STYLES[label]
    return (
        f'<span class="sg-sev-badge" '
        f'style="background:{st_cfg["bg"]};border-color:{st_cfg["border"]};color:{st_cfg["text"]};">'
        f'<span class="sg-sev-dot" style="background:{st_cfg["dot"]};"></span>'
        f'{_e(label)}</span>'
    )


def render_inspector_card(row: pd.Series) -> None:
    """Render a detailed diagnostic card for a single flagged record."""
    sev = str(row.get("severity", "Medium") or "Medium")
    cat = str(row.get("anomaly_category", "Unknown"))
    risk = float(row.get("risk_score", 0.0))
    ascore = float(row.get("anomaly_score", 0.0))
    sku = str(row.get("sku", ""))
    channel = str(row.get("channel", ""))
    explanation = str(row.get("explanation", ""))

    sev_pill = severity_badge_html(sev)
    _render_html(
        st,
        f"""
        <div class="sg-inspector-banner">
            <div class="sg-inspector-row">
                <div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">
                    {sev_pill}
                    <span class="sg-pill sg-pill-blue"><strong>{_e(cat)}</strong></span>
                    <span class="sg-pill">SKU: <strong>{_e(sku)}</strong></span>
                    <span class="sg-pill">Channel: <strong>{_e(channel)}</strong></span>
                </div>
                <div class="sg-pill-row">
                    <span class="sg-pill">Risk Score: <strong>{risk:.1f} / 100</strong></span>
                    <span class="sg-pill">Model Score: <strong>{ascore:.4f}</strong></span>
                </div>
            </div>
            <div class="sg-inspector-exp">{_e(explanation)}</div>
        </div>
        """,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    price_delta = float(row["channel_price"] - row["master_price"])
    stock_delta = float(row["channel_stock"] - row["master_stock"])
    _render_html(
        c1,
        f"""
        <div class="sg-cmp-card">
            <div class="sg-cmp-title">Channel vs Master Price</div>
            <div class="sg-cmp-val">${row["channel_price"]:,.2f}</div>
            <div class="sg-cmp-sub">
                Master: <strong>${row["master_price"]:,.2f}</strong> &middot;
                Diff: <strong>{price_delta:+,.2f}</strong> ({row["price_diff_pct"]:+.1%})
            </div>
        </div>
        """,
    )
    _render_html(
        c2,
        f"""
        <div class="sg-cmp-card">
            <div class="sg-cmp-title">Channel vs Master Stock</div>
            <div class="sg-cmp-val">{row["channel_stock"]:,.0f} units</div>
            <div class="sg-cmp-sub">
                Master: <strong>{row["master_stock"]:,.0f}</strong> &middot;
                Gap: <strong>{stock_delta:+,.0f}</strong> (Unexplained: {row["stock_residual"]:+,.1f})
            </div>
        </div>
        """,
    )
    _render_html(
        c3,
        f"""
        <div class="sg-cmp-card">
            <div class="sg-cmp-title">Synchronisation Delay</div>
            <div class="sg-cmp-val">{row["sync_delay_minutes"]:,.0f} min</div>
            <div class="sg-cmp-sub">
                Cadence multiple: <strong>{row["delay_ratio"]:.1f}x</strong> normal sync interval
            </div>
        </div>
        """,
    )


def render_validation_checklist(
    columns_present: set[str],
    required_cols: list[str],
    optional_cols: list[str],
) -> None:
    """Display required and optional CSV column status pills for uploaded files."""
    req_pills = []
    for col in required_cols:
        if col in columns_present:
            req_pills.append(f'<span class="sg-check-ok">&#10003; {_e(col)}</span>')
        else:
            req_pills.append(f'<span class="sg-check-miss">&#10007; {_e(col)}</span>')

    opt_pills = []
    for col in optional_cols:
        if col in columns_present:
            opt_pills.append(f'<span class="sg-check-ok">&#10003; {_e(col)}</span>')
        else:
            opt_pills.append(f'<span class="sg-check-opt">{_e(col)}</span>')

    _render_html(
        st,
        f"""
        <div style="margin-top:0.5rem;">
            <div style="font-size:0.78rem;font-weight:700;color:{NAVY};margin-bottom:0.25rem;">
                Required columns
            </div>
            <div class="sg-check-grid">{''.join(req_pills)}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{NAVY};margin:0.75rem 0 0.25rem 0;">
                Optional columns
            </div>
            <div class="sg-check-grid">{''.join(opt_pills)}</div>
        </div>
        """,
    )
