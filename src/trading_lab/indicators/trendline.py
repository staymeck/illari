"""Analytic geometry applied to market structure.

Structure (`structure.py`) gives swing highs/lows as loose points. Here we
fit a **least-squares line** (classic analytic geometry: y = mx + b) over
the most recent swings, to get a real support/resistance line with a
slope (trend angle) and to measure the current price's distance to that
line — instead of only looking at the last isolated swing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fits y = m*x + b by least squares. Requires at least 2 points."""
    if len(xs) < 2:
        raise ValueError("At least 2 points are needed to fit a line")
    slope, intercept = np.polyfit(xs, ys, deg=1)
    return float(slope), float(intercept)


def line_value_at(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def distance_to_line(slope: float, intercept: float, x: float, y: float) -> float:
    """Vertical distance (in price units) between the point (x,y) and the
    line. Vertical distance is used (not the Euclidean perpendicular one)
    because it's the one with financial meaning: how far price still has
    to go to touch the support/resistance line."""
    return y - line_value_at(slope, intercept, x)


def trend_angle_degrees(slope: float) -> float:
    """Converts the slope (price per candle) into an angle in degrees,
    purely as an interpretable way to compare how steep the trend is (it
    has no real physical units, it's a monotonic transform of the slope
    via arctan)."""
    return math.degrees(math.atan(slope))


def _fit_from_swings(
    df_with_swings: pd.DataFrame, as_of_index: int, kind: str, min_points: int, max_points: int
) -> tuple[float, float] | None:
    """Fits a line over the last `max_points` swings (high or low) already
    confirmed up to `as_of_index` (no lookahead, same criterion as
    `structure.last_swing_range`). Returns None if there aren't enough."""
    col = "swing_high" if kind == "high" else "swing_low"
    value_col = "high" if kind == "high" else "low"

    visible = df_with_swings[
        df_with_swings[col] & (df_with_swings["confirmed_at_index"] <= as_of_index)
    ]
    if len(visible) < min_points:
        return None

    recent = visible.iloc[-max_points:]
    xs = recent.index.to_numpy(dtype=float)
    ys = recent[value_col].to_numpy(dtype=float)
    return fit_line(list(xs), list(ys))


def attach_trendline_features(
    structure_df_with_swings: pd.DataFrame, min_points: int = 3, max_points: int = 6
) -> pd.DataFrame:
    """Adds, candle by candle, the support line (over swing lows) and the
    resistance line (over swing highs) fitted with the swings already
    confirmed up to that candle — no lookahead. New columns:
    `support_line_value`, `support_angle`, `dist_to_support`,
    `resistance_line_value`, `resistance_angle`, `dist_to_resistance`.
    """
    out = structure_df_with_swings.reset_index(drop=True)
    n = len(out)

    support_value = [None] * n
    support_angle = [None] * n
    dist_support = [None] * n
    resistance_value = [None] * n
    resistance_angle = [None] * n
    dist_resistance = [None] * n

    for i in range(n):
        close = out.at[i, "close"]

        support_fit = _fit_from_swings(out, i, "low", min_points, max_points)
        if support_fit is not None:
            slope, intercept = support_fit
            val = line_value_at(slope, intercept, i)
            support_value[i] = val
            support_angle[i] = trend_angle_degrees(slope)
            dist_support[i] = distance_to_line(slope, intercept, i, close)

        resistance_fit = _fit_from_swings(out, i, "high", min_points, max_points)
        if resistance_fit is not None:
            slope, intercept = resistance_fit
            val = line_value_at(slope, intercept, i)
            resistance_value[i] = val
            resistance_angle[i] = trend_angle_degrees(slope)
            dist_resistance[i] = distance_to_line(slope, intercept, i, close)

    out["support_line_value"] = support_value
    out["support_angle"] = support_angle
    out["dist_to_support"] = dist_support
    out["resistance_line_value"] = resistance_value
    out["resistance_angle"] = resistance_angle
    out["dist_to_resistance"] = dist_resistance
    return out
