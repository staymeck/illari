"""Confirmation piece: the trend has real strength (ADX above a threshold),
not just a technical direction. Wraps src.analysis.adx.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.adx import adx
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("adx_strength")
def adx_strength(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    period = params.get("period", 14)
    min_adx = params.get("min_adx", 25.0)

    last_adx = adx(ctx.price_window, period=period).iloc[-1]
    if pd.isna(last_adx) or last_adx < min_adx:
        return None
    return ConfirmationResult(name="adx_strength", extras={"adx": float(last_adx)})
