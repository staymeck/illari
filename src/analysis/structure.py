"""Market structure: swing highs/lows, trend, and support/resistance levels —
computed with objective code rules, not "eyeballed".

Reference: docs/PLAN.md — "Technical analysis engine components", item 1;
and resources/12_claves_analisis_tecnico.pdf (Dow Theory, support and resistance).
"""
from __future__ import annotations

import pandas as pd

# A swing point (pivot) requires `order` bars lower/higher on each side to be
# confirmed — this avoids marking a single bar of noise as a real pivot.
_DEFAULT_ORDER = 3


def find_swing_points(df: pd.DataFrame, order: int = _DEFAULT_ORDER) -> pd.DataFrame:
    """Marks confirmed swing high/low pivots (simple fractal: the pivot is the
    extreme within a `2*order + 1` bar window centered on it).

    Returns the same DataFrame with two new boolean columns:
    `is_swing_high`, `is_swing_low`.
    """
    highs = df["high"]
    lows = df["low"]
    window = 2 * order + 1

    is_swing_high = highs.rolling(window, center=True).apply(
        lambda w: w.iloc[order] == w.max(), raw=False
    ).fillna(0).astype(bool)
    is_swing_low = lows.rolling(window, center=True).apply(
        lambda w: w.iloc[order] == w.min(), raw=False
    ).fillna(0).astype(bool)

    out = df.copy()
    out["is_swing_high"] = is_swing_high
    out["is_swing_low"] = is_swing_low
    return out


def classify_trend(df: pd.DataFrame, order: int = _DEFAULT_ORDER, lookback_swings: int = 2) -> str:
    """Classifies the trend per Dow Theory: compares the last `lookback_swings`
    confirmed highs and lows.

    - "uptrend": rising highs and rising lows
    - "downtrend": falling highs and falling lows
    - "sideways": not enough swings, or they are not consistent
    """
    marked = find_swing_points(df, order=order)
    highs = marked.loc[marked["is_swing_high"], "high"].tail(lookback_swings)
    lows = marked.loc[marked["is_swing_low"], "low"].tail(lookback_swings)

    if len(highs) < lookback_swings or len(lows) < lookback_swings:
        return "sideways"

    highs_rising = highs.is_monotonic_increasing
    lows_rising = lows.is_monotonic_increasing
    highs_falling = highs.is_monotonic_decreasing
    lows_falling = lows.is_monotonic_decreasing

    if highs_rising and lows_rising:
        return "uptrend"
    if highs_falling and lows_falling:
        return "downtrend"
    return "sideways"


def support_resistance_levels(
    df: pd.DataFrame, order: int = _DEFAULT_ORDER, tolerance_pct: float = 0.5
) -> dict[str, list[float]]:
    """Groups swing lows into support levels and swing highs into resistance
    levels, merging pivots that fall within `tolerance_pct`% of each other (so
    we don't report 10 "levels" that are in practice the same one)."""
    marked = find_swing_points(df, order=order)

    def _cluster(values: pd.Series) -> list[float]:
        levels: list[float] = []
        for value in sorted(values.tolist()):
            if levels and abs(value - levels[-1]) / levels[-1] * 100 <= tolerance_pct:
                levels[-1] = (levels[-1] + value) / 2  # merge into the nearby level
            else:
                levels.append(value)
        return levels

    supports = _cluster(marked.loc[marked["is_swing_low"], "low"])
    resistances = _cluster(marked.loc[marked["is_swing_high"], "high"])
    return {"support": supports, "resistance": resistances}
