"""Confirmation piece: only take the entry during specific market sessions
(UTC), wrapping config.markets.session_for_hour.

This is the "hora de entrada al mercado" analysis variable from the very
start of docs/PLAN.md, turned into an actual entry filter rather than just a
post-hoc breakdown in the report — pass e.g. `sessions: [london, new_york]`
to only trade during those hours.

Note: this checks the *signal* bar's hour, not the actual fill (which is one
bar later, at the next candle's open — see src/backtest/engine.py). On
sub-hourly timeframes the two can occasionally fall in different sessions
right at a boundary; close enough for this filter's purpose, and avoids
threading the next bar's timestamp through EvalContext for a one-bar edge
case.
"""
from __future__ import annotations

from config.markets import session_for_hour
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("session_filter")
def session_filter(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    allowed_sessions = params.get("sessions")
    if not allowed_sessions:
        raise ValueError("session_filter requires a non-empty 'sessions' param, e.g. [london, new_york]")

    entry_hour_utc = ctx.price_window["timestamp"].iloc[-1].hour
    session = session_for_hour(entry_hour_utc)
    if session not in allowed_sessions:
        return None
    return ConfirmationResult(name="session_filter", extras={"session": session})
