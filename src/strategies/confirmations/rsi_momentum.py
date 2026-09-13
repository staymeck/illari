"""Confirmation piece: RSI showing bullish momentum (above a threshold) —
wraps src.analysis.momentum.rsi.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.momentum import rsi
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("rsi_momentum")
def rsi_momentum(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    period = params.get("period", 14)
    min_rsi = params.get("min_rsi", 50.0)

    values = rsi(ctx.price_window["close"], period=period)
    last_rsi = values.iloc[-1]
    if pd.isna(last_rsi) or last_rsi < min_rsi:
        return None
    return ConfirmationResult(name="rsi_momentum", extras={"rsi": float(last_rsi)})
