"""Analytic geometry applied to market structure: a least-squares line
(y = m*x + b) fit over the most recent CONFIRMED swing highs/lows, instead
of only comparing the single latest one (see structure.py's
support_resistance_levels, which reports flat clustered levels with no
sense of slope). Gives a real trendline with a slope/angle and a directly
interpretable "how far price still has to go to touch it" distance — a
genuinely different geometric idea from anything else in this catalog.

Rescued and reimplemented from a colleague's exploratory branch
(probabilities-to-into-trade, David A — not part of this project's own
git history) against THIS project's own find_swing_points/EvalContext
conventions, not that branch's own swing detector.

Takes `marked` (a DataFrame already carrying is_swing_high/is_swing_low,
e.g. ctx.marked_window) as given — same convention as structure.py's own
functions: no-lookahead safety is the CALLER's responsibility (the engine
and live lab always hand pieces an already safely-windowed marked
DataFrame — see engine.py's _confirmed_pivots_as_of), not re-derived here.
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
    """Vertical distance (price units) from (x, y) to the line — vertical,
    not perpendicular, because it's the financially meaningful one: how
    far price still has to move to touch the support/resistance line."""
    return y - line_value_at(slope, intercept, x)


def trend_angle_degrees(slope: float) -> float:
    """Slope (price per bar) as an angle in degrees — purely an
    interpretable, monotonic transform (arctan) to compare how steep
    different trendlines are; it has no real physical unit of its own."""
    return math.degrees(math.atan(slope))


def fit_swing_trendline(
    marked: pd.DataFrame, kind: str, min_points: int = 3, max_points: int = 6
) -> tuple[float, float] | None:
    """Fits a line over the last `max_points` swing highs (kind="high") or
    lows (kind="low") in `marked` (an is_swing_high/is_swing_low-tagged
    DataFrame, e.g. ctx.marked_window — already confirmed-as-of-now by the
    caller). Returns None if fewer than `min_points` swings are available
    yet. Uses `marked`'s own positional index as x (bars are evenly
    spaced, so this is a time axis up to a fixed scale) and the swing's
    own high/low price as y."""
    col = "is_swing_high" if kind == "high" else "is_swing_low"
    value_col = "high" if kind == "high" else "low"
    if col not in marked.columns or marked.empty:
        return None

    swings = marked.loc[marked[col]]
    if len(swings) < min_points:
        return None

    recent = swings.iloc[-max_points:]
    xs = recent.index.to_numpy(dtype=float)
    ys = recent[value_col].to_numpy(dtype=float)
    return fit_line(list(xs), list(ys))


def trendline_distance(
    marked: pd.DataFrame,
    current_index: float,
    current_price: float,
    kind: str,
    min_points: int = 3,
    max_points: int = 6,
) -> float | None:
    """Fits the swing trendline (see fit_swing_trendline) and returns the
    vertical distance from `current_price` (at `current_index`, in the
    same positional-index space as `marked`) to it — positive means price
    sits above the line, negative below. None when there isn't a fitted
    line yet."""
    fit = fit_swing_trendline(marked, kind, min_points, max_points)
    if fit is None:
        return None
    slope, intercept = fit
    return distance_to_line(slope, intercept, current_index, current_price)
