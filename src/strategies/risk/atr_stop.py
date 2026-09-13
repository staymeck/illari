"""Risk piece: a stop at N multiples of ATR from entry, instead of a fixed
percentage of the setup's reference level (see risk/fixed_pct.py).

Why this matters (see docs/PLAN.md — the 5m-vs-1h comparison): a fixed-%
stop is the same width regardless of how much a market actually moves in
that timeframe, which is exactly what made 5m unworkable (noise-sized moves
next to a cost-sized stop). An ATR stop scales with the market's own recent
volatility instead.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.volatility import atr
from src.strategies.registry import RISK_STOPS
from src.strategies.types import EvalContext, SetupResult


@RISK_STOPS.register("atr_multiple")
def atr_stop(entry_price: float, setup: SetupResult, ctx: EvalContext, params: dict) -> float:
    period = params.get("period", 14)
    multiple = params.get("multiple", 1.5)

    last_atr = atr(ctx.price_window, period=period).iloc[-1]
    if pd.isna(last_atr):
        # Not enough data to size a real stop yet — fall back to a tight,
        # conservative default (0.5% below entry) rather than no stop at all.
        return entry_price * 0.995

    return entry_price - multiple * last_atr
