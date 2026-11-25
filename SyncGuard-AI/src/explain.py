"""Human-readable explanations for flagged records.

Isolation Forest tells us *that* a record is unusual, not *why*.  The
:class:`Explainer` bridges that gap: tolerances are learned from the training
records the model considered normal, and every flagged record is checked
against them to produce plain-English reason codes (price, inventory, delay,
invalid data).  If none of the reason codes applies, the explanation falls back
to the features that deviate most (in robust-scaled units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import FEATURE_LABELS


def _fmt_minutes(m: float) -> str:
    if m < 120:
        return f"{m:.0f} min"
    if m < 60 * 48:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} days"


class Explainer:
    def __init__(self, price_tol: float = 0.04, stock_tol: float = 5.0, delay_tol: float = 2.5):
        self.price_tol = float(price_tol)
        self.stock_tol = float(stock_tol)
        self.delay_tol = float(delay_tol)

    @classmethod
    def from_training(cls, feats_normal: pd.DataFrame) -> "Explainer":
        """Learn tolerances from rows the model itself considered normal."""
        price_tol = max(float(np.percentile(feats_normal["abs_price_dev_pct"], 99)) * 1.25, 0.03)
        stock_tol = max(float(np.percentile(np.abs(feats_normal["stock_residual"]), 99)) * 1.25, 3.0)
        delay_tol = max(float(np.percentile(feats_normal["delay_ratio"], 99)) * 1.10, 2.5)
        return cls(price_tol, stock_tol, delay_tol)

    def tolerances(self) -> dict:
        return {
            "price_deviation_tolerance_pct": round(self.price_tol * 100, 2),
            "stock_gap_tolerance_units": round(self.stock_tol, 1),
            "delay_tolerance_x_cadence": round(self.delay_tol, 2),
        }

    # ------------------------------------------------------------------
    def explain(self, raw: pd.DataFrame, feats: pd.DataFrame, scaled: np.ndarray,
                feature_names: list, mask: np.ndarray):
        """Return (explanations, categories) lists aligned with ``raw`` rows."""
        n = len(raw)
        explanations = [""] * n
        categories = [""] * n
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return explanations, categories

        ch = raw["channel"].to_numpy()
        mp = raw["master_price"].to_numpy(dtype=float)
        cp = raw["channel_price"].to_numpy(dtype=float)
        ms = raw["master_stock"].to_numpy(dtype=float)
        cs = raw["channel_stock"].to_numpy(dtype=float)
        delay = raw["sync_delay_minutes"].to_numpy(dtype=float)

        dev = feats["abs_price_dev_pct"].to_numpy()
        pct = feats["price_diff_pct"].to_numpy()
        exp_pct = feats["expected_price_diff_pct"].to_numpy()
        sdiff = feats["stock_diff"].to_numpy()
        resid = feats["stock_residual"].to_numpy()
        lag = feats["expected_lag_units"].to_numpy()
        dratio = feats["delay_ratio"].to_numpy()
        fexp = feats["expected_sync_minutes"].to_numpy()
        stockout = feats["stockout_mismatch"].to_numpy()

        for i in idx:
            reasons, kinds = [], []

            # ---- price ---------------------------------------------------
            if cp[i] <= 0 or mp[i] <= 0:
                reasons.append(f"Invalid price: channel price is {cp[i]:,.2f} (master {mp[i]:,.2f}).")
                kinds.append("Invalid Data")
            elif dev[i] >= self.price_tol:
                expected_price = mp[i] * (1 + exp_pct[i])
                ratio = cp[i] / expected_price if expected_price > 0 else 1.0
                hint = ""
                if ratio >= 8 or ratio <= 0.125:
                    hint = " Looks like a decimal-place / unit error."
                reasons.append(
                    f"Price mismatch: {ch[i]} lists {cp[i]:,.2f} vs master {mp[i]:,.2f} "
                    f"({pct[i]:+.1%}); normal for {ch[i]} is {exp_pct[i]:+.1%}.{hint}"
                )
                kinds.append("Price Mismatch")

            # ---- inventory -----------------------------------------------
            if cs[i] < 0 or ms[i] < 0:
                reasons.append(f"Invalid stock: channel shows {cs[i]:.0f} units (master {ms[i]:.0f}).")
                kinds.append("Invalid Data")
            elif stockout[i] == 1:
                reasons.append(
                    f"False stock-out: channel shows 0 units while master has {ms[i]:.0f} "
                    f"(lost-sales risk)."
                )
                kinds.append("Inventory Mismatch")
            elif resid[i] > self.stock_tol:
                reasons.append(
                    f"Overselling risk: channel shows {sdiff[i]:+.0f} units vs master "
                    f"(only ~{lag[i]:.1f} explained by sales since last sync)."
                )
                kinds.append("Inventory Mismatch")
            elif resid[i] < -self.stock_tol:
                reasons.append(
                    f"Understated stock: channel shows {abs(sdiff[i]):.0f} fewer units than master "
                    f"({cs[i]:.0f} vs {ms[i]:.0f}), risking lost sales."
                )
                kinds.append("Inventory Mismatch")

            # ---- sync delay ----------------------------------------------
            if dratio[i] >= self.delay_tol:
                reasons.append(
                    f"Sync delay: last sync {_fmt_minutes(delay[i])} ago, "
                    f"{dratio[i]:.1f}x the normal {fexp[i]:.0f}-min cadence."
                )
                kinds.append("Sync Delay")

            # ---- fallback ------------------------------------------------
            if not reasons:
                order = np.argsort(-np.abs(scaled[i]))[:2]
                names = []
                for j in order:
                    lbl = FEATURE_LABELS.get(feature_names[j], feature_names[j])
                    if lbl not in names:
                        names.append(lbl)
                reasons.append(
                    "Unusual combination of values: no single rule breached, but the record is "
                    "isolated quickly by the model (largest deviations: " + ", ".join(names) + ")."
                )
                kinds.append("Statistical Outlier")

            uniq = list(dict.fromkeys(kinds))
            categories[i] = uniq[0] if len(uniq) == 1 else "Multiple Issues"
            explanations[i] = " ".join(reasons)

        return explanations, categories
