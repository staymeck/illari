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


def nearest_support_below(price: float, levels: list[float]) -> float | None:
    """The highest level in `levels` that sits below `price` — i.e. the
    next real structural floor price would have to fall through — or None
    if every level is at or above `price`.

    Deliberately generic: `levels` can be support pivots, Fibonacci
    retracement prices, or both combined — this just picks the nearest one
    down, wherever it came from. Used by risk/structural_stop.py to place a
    stop at a real level instead of an arbitrary fixed percentage."""
    candidates = [level for level in levels if level < price]
    return max(candidates) if candidates else None


def nearest_resistance_above(price: float, levels: list[float]) -> float | None:
    """Mirror image of nearest_support_below: the lowest level in `levels`
    that sits above `price` — the next real structural ceiling price would
    have to break through — or None if every level is at or below `price`.
    Used by risk/structural_target.py to place a take-profit at a real
    level instead of an arbitrary risk-multiple."""
    candidates = [level for level in levels if level > price]
    return min(candidates) if candidates else None


def confluence_count(level: float, *level_groups: list[float], tolerance_pct: float = 0.5) -> int:
    """How many of the given `level_groups` (e.g. support pivots, Fibonacci
    levels — pass each as a separate list) have at least one level within
    `tolerance_pct`% of `level`. A rough, purely geometric proxy for how
    much independent structural agreement — "liquidity" in the sense of
    real prior interest, not order-book depth we don't have — sits at that
    price: a level only one source points to is easier for price to cut
    through than one several sources agree on."""
    count = 0
    for levels in level_groups:
        if any(abs(other - level) / level * 100 <= tolerance_pct for other in levels):
            count += 1
    return count


def nearest_confluent_level(
    price: float,
    candidates: list[float],
    confluence_groups: tuple[list[float], ...],
    direction: str,
    min_confluence: int = 1,
    tolerance_pct: float = 0.5,
) -> float | None:
    """Like nearest_support_below / nearest_resistance_above, but walks
    outward past any candidate that doesn't have at least `min_confluence`
    independent `confluence_groups` agreeing near it (see confluence_count)
    — skipping a noise-level, single-source level (e.g. a small bump inside
    a retracement's own back-and-forth) to find one with real backing,
    instead of anchoring to whichever happens to be nearest.

    `direction`: "below" walks downward (nearest first), "above" walks
    upward (nearest first). `min_confluence=1` never filters anything — a
    candidate drawn from one of `confluence_groups` always self-matches
    that group — so it reproduces nearest_support_below /
    nearest_resistance_above exactly; `min_confluence=2` requires genuine
    cross-confirmation from a second, independent source."""
    if direction == "below":
        ordered = sorted((c for c in candidates if c < price), reverse=True)
    elif direction == "above":
        ordered = sorted(c for c in candidates if c > price)
    else:
        raise ValueError(f"direction must be 'below' or 'above', got {direction!r}")

    for level in ordered:
        if confluence_count(level, *confluence_groups, tolerance_pct=tolerance_pct) >= min_confluence:
            return level
    return None
