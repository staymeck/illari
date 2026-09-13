"""Context piece: Dow-theory trend classification.

Wraps the existing src.analysis.structure.classify_trend — see docs/PLAN.md,
"Config-driven strategy catalog".
"""
from __future__ import annotations

from src.analysis.structure import classify_trend
from src.strategies.registry import CONTEXT
from src.strategies.types import EvalContext


@CONTEXT.register("dow_trend")
def dow_trend(ctx: EvalContext, params: dict) -> str:
    order = params.get("order", 3)
    lookback_swings = params.get("lookback_swings", 2)
    return classify_trend(
        ctx.marked_window, order=order, lookback_swings=lookback_swings, marked=ctx.marked_window
    )
