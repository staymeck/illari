"""Context piece: moving-average cross/slope trend classification.

An alternative to dow_trend (swing-point based) — wraps
src.analysis.moving_average.ma_cross_trend. See docs/PLAN.md, "Config-driven
strategy catalog".
"""
from __future__ import annotations

from src.analysis.moving_average import ma_cross_trend
from src.strategies.registry import CONTEXT
from src.strategies.types import EvalContext


@CONTEXT.register("ma_trend")
def ma_trend(ctx: EvalContext, params: dict) -> str:
    fast = params.get("fast", 20)
    slow = params.get("slow", 50)
    method = params.get("method", "ema")
    return ma_cross_trend(ctx.price_window, fast=fast, slow=slow, method=method)
