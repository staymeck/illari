"""Confirmation piece: is price near the least-squares support trendline
fit over recent swing lows (see src.analysis.trendline)? Distinct from the
flat, clustered support levels risk/structural_stop.py anchors to: this is
a line with a slope, so "near" tracks a moving target (a rising or falling
support) instead of a fixed price — analytic geometry over market
structure, rescued from a colleague's exploratory branch
(probabilities-to-into-trade, David A).
"""
from __future__ import annotations

import pandas as pd

from src.analysis.trendline import trendline_distance
from src.analysis.volatility import atr
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("trendline")
def trendline_confirmation(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    """Passes when price sits within `proximity_atr_mult` (default 1.0)
    ATRs of the fitted support trendline over the last `min_points`-
    `max_points` confirmed swing lows in `ctx.marked_window`. Long-only for
    now (checks the support line), mirrors the rest of this catalog."""
    min_points = params.get("min_points", 3)
    max_points = params.get("max_points", 6)
    proximity_atr_mult = params.get("proximity_atr_mult", 1.0)
    atr_period = params.get("atr_period", 14)

    df = ctx.price_window
    if df.empty or ctx.marked_window is None or ctx.marked_window.empty:
        return None

    current_index = float(df.index[-1])
    current_price = float(df["close"].iloc[-1])
    distance = trendline_distance(
        ctx.marked_window, current_index, current_price, kind="low", min_points=min_points, max_points=max_points
    )
    if distance is None:
        return None

    last_atr = atr(df, period=atr_period).iloc[-1]
    if pd.isna(last_atr) or last_atr <= 0:
        return None

    if abs(distance) > proximity_atr_mult * last_atr:
        return None
    return ConfirmationResult(name="trendline", extras={"trendline_distance": round(float(distance), 4)})
