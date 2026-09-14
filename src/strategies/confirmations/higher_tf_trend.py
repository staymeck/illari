"""Confirmation pieces: a higher timeframe (e.g. 1d while trading 1h) must
also be in the required trend — Kaufman, Trading Systems and Methods,
ch. 19 "Multiple Time Frames" (Elder's Triple Screen, Krausz, Pring's KST
all share this same core idea: don't take a lower-timeframe signal against
the higher-timeframe trend).

Two registered pieces, not one: `higher_tf_trend` reads `ctx.higher_tf_window`
(populated by `run_backtest`'s `higher_tf_df`) and `higher_tf_trend_2` reads
`ctx.higher_tf_window_2` (populated by `higher_tf_df_2`) — e.g. 1d and 4h
as two INDEPENDENT confirmations while trading 1h, per Bitácora Illari's
staged 4H-filter experiment. Same logic, same params, different window —
use both in one strategy's `confirmations` list to require agreement
across three timeframes at once.

Either fails closed (returns None, i.e. "can't confirm") rather than
silently passing when its window wasn't supplied.
"""
from __future__ import annotations

from src.analysis.structure import classify_trend
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


def _higher_tf_trend_check(window, params: dict, extra_key: str) -> ConfirmationResult | None:
    if window is None or window.empty:
        return None  # no higher-timeframe data available -> can't confirm

    required = params.get("required", "uptrend")
    order = params.get("order", 3)
    lookback_swings = params.get("lookback_swings", 2)

    trend = classify_trend(window, order=order, lookback_swings=lookback_swings)
    if trend != required:
        return None
    return ConfirmationResult(name=extra_key, extras={extra_key: trend})


@CONFIRMATIONS.register("higher_tf_trend")
def higher_tf_trend(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    return _higher_tf_trend_check(ctx.higher_tf_window, params, "higher_tf_trend")


@CONFIRMATIONS.register("higher_tf_trend_2")
def higher_tf_trend_2(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    return _higher_tf_trend_check(ctx.higher_tf_window_2, params, "higher_tf_trend_2")
