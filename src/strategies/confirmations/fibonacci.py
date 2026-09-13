"""Confirmation piece: price near a Fibonacci retracement of the latest
confirmed up-leg.

Wraps the existing src.analysis.fibonacci — see docs/PLAN.md, "Config-driven
strategy catalog".
"""
from __future__ import annotations

from src.analysis.fibonacci import is_near_confluence, latest_up_leg
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("fibonacci")
def fibonacci_confluence(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    order = params.get("order", 3)
    tolerance_pct = params.get("tolerance_pct", 1.0)

    leg = latest_up_leg(ctx.marked_window, order=order, marked=ctx.marked_window)
    if leg is None:
        return None

    last_close = ctx.price_window["close"].iloc[-1]
    confluence = is_near_confluence(last_close, leg["low"], leg["high"], tolerance_pct)
    if confluence is None:
        return None

    fib_ratio, fib_level = confluence
    return ConfirmationResult(name="fibonacci", extras={"fib_ratio": fib_ratio, "fib_level": fib_level})
