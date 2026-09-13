"""Fibonacci retracement levels, used strictly as a confluence zone alongside
market structure — never as a standalone signal (see docs/PLAN.md,
"Technical analysis engine components").
"""
from __future__ import annotations

import pandas as pd

from src.analysis.structure import find_swing_points

# Standard retracement ratios.
RETRACEMENT_RATIOS: list[float] = [0.236, 0.382, 0.5, 0.618, 0.786]


def retracement_levels(swing_low: float, swing_high: float) -> dict[float, float]:
    """Retracement price levels of an up-leg from `swing_low` to `swing_high`,
    keyed by ratio (e.g. `0.618` -> the price of the 61.8% retracement)."""
    if swing_high <= swing_low:
        raise ValueError("swing_high must be greater than swing_low")
    leg_range = swing_high - swing_low
    return {ratio: swing_high - leg_range * ratio for ratio in RETRACEMENT_RATIOS}


def latest_up_leg(df: pd.DataFrame, order: int = 3, marked: pd.DataFrame | None = None) -> dict | None:
    """Finds the most recently completed up-leg: a confirmed swing low
    followed (later in time) by a confirmed swing high. Returns None when the
    most recent confirmed pivot is a low instead of a high — meaning price is
    currently making new highs with no pullback yet to retrace against.

    If `marked` is given (already carrying `is_swing_high`/`is_swing_low`,
    e.g. computed once upfront for a whole backtest run — see
    src/backtest/engine.py), it's used as-is instead of recomputing it."""
    marked = marked if marked is not None else find_swing_points(df, order=order)
    highs = marked.loc[marked["is_swing_high"]]
    lows = marked.loc[marked["is_swing_low"]]

    if highs.empty or lows.empty:
        return None

    last_high_idx = highs.index[-1]
    last_low_idx = lows.index[-1]

    if last_low_idx >= last_high_idx:
        return None  # most recent pivot is a low: no completed up-leg yet

    candidate_lows = lows.loc[lows.index < last_high_idx]
    if candidate_lows.empty:
        return None
    low_idx = candidate_lows.index[-1]

    return {
        "low": float(marked.loc[low_idx, "low"]),
        "low_idx": int(low_idx),
        "high": float(marked.loc[last_high_idx, "high"]),
        "high_idx": int(last_high_idx),
    }


def nearest_level(price: float, levels: dict[float, float]) -> tuple[float, float]:
    """Returns `(ratio, level_price)` of the retracement level closest to `price`."""
    ratio, level_price = min(levels.items(), key=lambda item: abs(item[1] - price))
    return ratio, level_price


def is_near_confluence(
    price: float, swing_low: float, swing_high: float, tolerance_pct: float = 0.5
) -> tuple[float, float] | None:
    """Checks whether `price` sits within `tolerance_pct`% of any standard
    retracement level of the leg `[swing_low, swing_high]`. Returns the
    matching `(ratio, level_price)`, or None if nothing is close enough."""
    levels = retracement_levels(swing_low, swing_high)
    ratio, level_price = nearest_level(price, levels)
    distance_pct = abs(price - level_price) / level_price * 100
    if distance_pct <= tolerance_pct:
        return ratio, level_price
    return None
