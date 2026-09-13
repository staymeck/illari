"""Setup piece: price touching or piercing the lower Bollinger Band — a
mean-reversion entry zone, the opposite philosophy from trend-following
setups like support_touch/breakout. Pair with `context.required: sideways`
(or `any`) — gating this behind an uptrend, as trend-following setups need,
would defeat the point.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.volatility import bollinger_bands
from src.strategies.registry import SETUPS
from src.strategies.types import EvalContext, SetupResult


@SETUPS.register("mean_reversion")
def mean_reversion(ctx: EvalContext, params: dict) -> SetupResult | None:
    window = params.get("window", 20)
    num_std = params.get("num_std", 2.0)

    price = ctx.price_window
    bands = bollinger_bands(price["close"], window=window, num_std=num_std)
    lower = bands["lower"].iloc[-1]
    mid = bands["mid"].iloc[-1]
    if pd.isna(lower):
        return None

    last_close = price["close"].iloc[-1]
    if last_close <= lower:
        return SetupResult(reference_level=lower, extras={"bollinger_mid": mid, "bollinger_lower": lower})
    return None
