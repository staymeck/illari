"""Opportunity scan: an "oracle" pass over raw price data, upstream of any
strategy rule.

The question this answers is different from everything else in the lab so
far: forget entry filters for a moment - how much genuine profit
opportunity actually exists in this data, at all? For every bar, look
forward over a holding horizon and measure the best possible favorable
move (Maximum Favorable Excursion, MFE) in each direction, regardless of
whether any of our strategies would have recognized the setup. This is a
diagnostic against which to judge our filtered strategies' rarity: are they
being too strict (throwing away real opportunities) or about right
(capturing most of what's realistically achievable once costs are
subtracted)?

Motivated by trend_pullback_htf firing only ~1 trade per market every few
months live - is that a sign the filters are broken, or a sign that
genuine, cost-clearing opportunities are simply this rare in this data?
See docs/PLAN.md / Bitácora Illari.

Deliberately NOT a trading signal: this uses real look-ahead by
construction (the best price achieved AFTER the entry bar). It cannot be
used to enter a live trade - only to describe the data.
"""
from __future__ import annotations

import pandas as pd


def compute_mfe(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Adds `mfe_long_pct` / `mfe_short_pct`: the best possible % move
    achievable entering at this bar's close and exiting at the best price
    within the following `horizon_bars` bars (excluding the entry bar
    itself - the earliest realistic exit is the next bar).

    Rows in the final `horizon_bars` of `df` don't have a full forward
    window and are dropped rather than reported with a truncated (and
    therefore understated) MFE.
    """
    def _forward_extreme(series: pd.Series, agg: str) -> pd.Series:
        shifted = series.shift(-1)
        reversed_ = shifted.iloc[::-1]
        rolled = reversed_.rolling(window=horizon_bars, min_periods=horizon_bars)
        rolled = rolled.max() if agg == "max" else rolled.min()
        return rolled.iloc[::-1]

    future_high_max = _forward_extreme(df["high"], "max")
    future_low_min = _forward_extreme(df["low"], "min")

    out = df.copy()
    out["mfe_long_pct"] = (future_high_max - df["close"]) / df["close"] * 100
    out["mfe_short_pct"] = (df["close"] - future_low_min) / df["close"] * 100
    return out.dropna(subset=["mfe_long_pct", "mfe_short_pct"])


def opportunity_rate(df: pd.DataFrame, horizon_bars: int, threshold_pct: float) -> dict:
    """Fraction of bars where a long, and separately a short, move of at
    least `threshold_pct` was available within `horizon_bars` - plus the
    magnitude distribution (mean/median MFE) so a nonzero rate can be read
    alongside how big those opportunities typically were."""
    mfe = compute_mfe(df, horizon_bars)
    n = len(mfe)
    if n == 0:
        return {
            "n_bars": 0, "long_opportunity_pct": 0.0, "short_opportunity_pct": 0.0,
            "either_opportunity_pct": 0.0, "mean_mfe_long_pct": 0.0, "median_mfe_long_pct": 0.0,
            "mean_mfe_short_pct": 0.0, "median_mfe_short_pct": 0.0,
        }

    long_hit = mfe["mfe_long_pct"] >= threshold_pct
    short_hit = mfe["mfe_short_pct"] >= threshold_pct

    return {
        "n_bars": n,
        "long_opportunity_pct": round(100 * long_hit.mean(), 2),
        "short_opportunity_pct": round(100 * short_hit.mean(), 2),
        "either_opportunity_pct": round(100 * (long_hit | short_hit).mean(), 2),
        "mean_mfe_long_pct": round(mfe["mfe_long_pct"].mean(), 3),
        "median_mfe_long_pct": round(mfe["mfe_long_pct"].median(), 3),
        "mean_mfe_short_pct": round(mfe["mfe_short_pct"].mean(), 3),
        "median_mfe_short_pct": round(mfe["mfe_short_pct"].median(), 3),
    }
