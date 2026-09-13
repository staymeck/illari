"""Confirmation piece: a bullish reversal candlestick pattern.

Wraps the existing src.analysis.candles.bullish_reversal_pattern — see
docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

from src.analysis.candles import bullish_reversal_pattern
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("candlestick")
def candlestick(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    pattern = bullish_reversal_pattern(ctx.price_window, idx=len(ctx.price_window) - 1)
    if pattern is None:
        return None
    return ConfirmationResult(name="candlestick", extras={"pattern": pattern})
