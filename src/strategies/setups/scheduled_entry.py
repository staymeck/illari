"""Setup piece: enter at a fixed hour every day, regardless of any market
condition — no trend, no support, no confirmation. Pair with
`context.required: any` (skip the trend gate entirely) and set
`style.max_holding_bars` to the exact number of bars between the entry hour
and the intended exit hour, so the scheduled exit lines up with a timeout.

Tests whether a specific time-of-day window carries a consistent directional
bias by itself — a narrow, explicit version of the seasonality idea from
Kaufman, Trading Systems and Methods, ch. 10 ("Seasonality and Calendar
Patterns"), and directly comparable to the random-entry Monte Carlo
benchmark (src.backtest.random_benchmark) since it produces roughly one
trade per day, every day, with no selection at all beyond the clock.

Only meaningful on timeframes where each hour appears once per day (1h or
coarser) — on finer timeframes (5m/30m) `entry_hour` alone doesn't pick a
single bar.
"""
from __future__ import annotations

from src.strategies.registry import SETUPS
from src.strategies.types import EvalContext, SetupResult


@SETUPS.register("scheduled_entry")
def scheduled_entry(ctx: EvalContext, params: dict) -> SetupResult | None:
    entry_hour = params.get("entry_hour", 3)
    last = ctx.price_window.iloc[-1]
    if last["timestamp"].hour != entry_hour:
        return None
    return SetupResult(reference_level=last["close"], extras={"scheduled_entry_hour": entry_hour})
