"""Confirmation piece: positive MACD histogram (bullish momentum) — wraps
src.analysis.momentum.macd.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.momentum import macd
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("macd_momentum")
def macd_momentum(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal = params.get("signal", 9)

    result = macd(ctx.price_window["close"], fast=fast, slow=slow, signal=signal)
    last_hist = result["histogram"].iloc[-1]
    if pd.isna(last_hist) or last_hist <= 0:
        return None
    return ConfirmationResult(name="macd_momentum", extras={"macd_histogram": float(last_hist)})
