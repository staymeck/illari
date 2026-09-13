"""Confirmation piece: the higher timeframe (e.g. 1d while trading 1h) must
also be in the required trend — Kaufman, Trading Systems and Methods,
ch. 19 "Multiple Time Frames" (Elder's Triple Screen, Krausz, Pring's KST
all share this same core idea: don't take a lower-timeframe signal against
the higher-timeframe trend).

Requires the engine to have been given `higher_tf_df` (see
src/backtest/engine.run_backtest) — without it, `ctx.higher_tf_window` is
None and this confirmation always fails closed (returns None, i.e. "can't
confirm") rather than silently passing.
"""
from __future__ import annotations

from src.analysis.structure import classify_trend
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("higher_tf_trend")
def higher_tf_trend(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    if ctx.higher_tf_window is None or ctx.higher_tf_window.empty:
        return None  # no higher-timeframe data available -> can't confirm

    required = params.get("required", "uptrend")
    order = params.get("order", 3)
    lookback_swings = params.get("lookback_swings", 2)

    trend = classify_trend(ctx.higher_tf_window, order=order, lookback_swings=lookback_swings)
    if trend != required:
        return None
    return ConfirmationResult(name="higher_tf_trend", extras={"higher_tf_trend": trend})
