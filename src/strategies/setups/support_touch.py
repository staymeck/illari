"""Setup piece: price touching a confirmed, clustered support level.

Wraps the existing src.analysis.structure.support_resistance_levels — see
docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

from src.analysis.structure import support_resistance_levels
from src.strategies.registry import SETUPS
from src.strategies.types import EvalContext, SetupResult


@SETUPS.register("support_touch")
def support_touch(ctx: EvalContext, params: dict) -> SetupResult | None:
    order = params.get("order", 3)
    tolerance_pct = params.get("tolerance_pct", 1.0)

    levels = support_resistance_levels(ctx.marked_window, order=order, marked=ctx.marked_window)
    supports = levels["support"]
    if not supports:
        return None

    last_close = ctx.price_window["close"].iloc[-1]
    for level in supports:
        distance_pct = abs(last_close - level) / level * 100
        if last_close >= level and distance_pct <= tolerance_pct:
            return SetupResult(reference_level=level, extras={"support_level": level})
    return None
