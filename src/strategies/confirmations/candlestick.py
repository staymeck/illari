"""Confirmation pieces: a bullish or bearish reversal candlestick pattern.

Wraps src.analysis.candles.bullish_reversal_pattern / bearish_reversal_pattern
(now covering all 18 patterns in
resources/18-Patrones-de-velas-que-debes-conocer.pdf — see docs/PLAN.md).
"bearish_candlestick" isn't wired into any strategy YAML yet: the current
catalog is long/uptrend-only, but it's registered for whenever a short-side
strategy exists.
"""
from __future__ import annotations

from src.analysis.candles import bearish_reversal_pattern, bullish_reversal_pattern
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("candlestick")
def candlestick(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    pattern = bullish_reversal_pattern(ctx.price_window, idx=len(ctx.price_window) - 1)
    if pattern is None:
        return None
    return ConfirmationResult(name="candlestick", extras={"pattern": pattern})


@CONFIRMATIONS.register("bearish_candlestick")
def bearish_candlestick(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    pattern = bearish_reversal_pattern(ctx.price_window, idx=len(ctx.price_window) - 1)
    if pattern is None:
        return None
    return ConfirmationResult(name="bearish_candlestick", extras={"pattern": pattern})
