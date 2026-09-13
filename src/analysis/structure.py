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

    Implementation note: this uses pandas' built-in vectorized
    `rolling().max()/.min()` (a C-level, O(n) sliding-window algorithm)
    instead of `rolling().apply()` with a Python lambda — the latter calls
    back into Python once per window and was, by measurement, the single
    biggest cost in the whole backtest engine. Both give identical results:
    a bar is a swing high iff it equals the max of its own centered window,
    which `rolling(..., center=True).max()` computes for every position at
    once.
    """
    window = 2 * order + 1

    rolling_high = df["high"].rolling(window, center=True).max()
    rolling_low = df["low"].rolling(window, center=True).min()

    out = df.copy()
    out["is_swing_high"] = df["high"].eq(rolling_high) & rolling_high.notna()
    out["is_swing_low"] = df["low"].eq(rolling_low) & rolling_low.notna()
    return out


def classify_trend(
    df: pd.DataFrame,
    order: int = _DEFAULT_ORDER,
    lookback_swings: int = 2,
    marked: pd.DataFrame | None = None,
) -> str:
    """Classifies the trend per Dow Theory: compares the last `lookback_swings`
    confirmed highs and lows.

    - "uptrend": rising highs and rising lows
    - "downtrend": falling highs and falling lows
    - "sideways": not enough swings, or they are not consistent

    If `marked` is given (a DataFrame already carrying `is_swing_high`/
    `is_swing_low`, e.g. computed once upfront for a whole backtest run —
    see src/backtest/engine.py), it's used as-is and `df`/`order` are only
    used to compute it when `marked` is omitted.
    """
    marked = marked if marked is not None else find_swing_points(df, order=order)
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
    df: pd.DataFrame,
    order: int = _DEFAULT_ORDER,
    tolerance_pct: float = 0.5,
    marked: pd.DataFrame | None = None,
) -> dict[str, list[float]]:
    """Groups swing lows into support levels and swing highs into resistance
    levels, merging pivots that fall within `tolerance_pct`% of each other (so
    we don't report 10 "levels" that are in practice the same one).

    See `classify_trend` for what passing `marked` does."""
    marked = marked if marked is not None else find_swing_points(df, order=order)

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
