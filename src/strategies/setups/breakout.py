"""Setup piece: price breaking above the highest high of the prior N bars
(Donchian-style breakout) — the opposite entry philosophy from
support_touch (chasing a new high instead of waiting for a pullback).

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

from src.strategies.registry import SETUPS
from src.strategies.types import EvalContext, SetupResult


@SETUPS.register("breakout")
def breakout(ctx: EvalContext, params: dict) -> SetupResult | None:
    lookback = params.get("lookback", 20)
    price = ctx.price_window
    if len(price) < lookback + 1:
        return None

    # Highest high of the `lookback` bars strictly before the current one —
    # the current bar's own high must never count toward its own breakout level.
    prior_high = price["high"].iloc[-(lookback + 1) : -1].max()
    last_close = price["close"].iloc[-1]

    if last_close > prior_high:
        return SetupResult(reference_level=prior_high, extras={"breakout_level": prior_high})
    return None
