"""Risk piece: an ATR stop whose multiple itself scales with the CURRENT
volatility's percentile within its own recent history, instead of one fixed
multiple for every market/timeframe.

Why: this session found `atr_multiple` (a single fixed multiple) needs
different tuning per market/timeframe to avoid hurting results (helped
BTC/1h, wrecked BTC/30m and PAXG/30m with the same multiple — see
docs/PLAN.md). Scaling the multiple by where volatility sits relative to
that market's OWN recent distribution (src.analysis.regime.
volatility_percentile) is market-relative by construction — the hope is it
generalizes without per-market retuning. This is "Version A" adaptation
(to a measured market condition), not "Version B" (to the strategy's own
past results) — see docs/PLAN.md.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.regime import volatility_percentile
from src.analysis.volatility import atr
from src.strategies.registry import RISK_STOPS
from src.strategies.types import EvalContext, SetupResult


@RISK_STOPS.register("regime_adaptive")
def regime_adaptive_stop(entry_price: float, setup: SetupResult, ctx: EvalContext, params: dict) -> float:
    period = params.get("period", 14)
    lookback = params.get("lookback", 100)
    base_multiple = params.get("base_multiple", 1.0)  # multiple used at the 0th volatility percentile
    max_multiple = params.get("max_multiple", 3.0)  # multiple used at the 100th volatility percentile

    last_atr = atr(ctx.price_window, period=period).iloc[-1]
    if pd.isna(last_atr):
        # Not enough data to size a real stop yet — same conservative
        # fallback as risk/atr_stop.py.
        return entry_price * 0.995

    pct = volatility_percentile(ctx.price_window, atr_period=period, lookback=lookback)
    if pd.isna(pct):
        pct = 50.0  # unknown -> assume the middle of the distribution

    multiple = base_multiple + (max_multiple - base_multiple) * (pct / 100)
    return entry_price - multiple * last_atr
