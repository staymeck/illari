"""Confirmation pieces reading a FINER-grained window than the traded
timeframe (e.g. 5m while trading 1h) — `ctx.lower_tf_window`, populated
only when the engine was given `lower_tf_df` (src/backtest/engine.py).

Stage 2 of the user's staged experiment protocol (Bitácora Illari): unlike
the intrabar-entry idea already tried and rejected
(src/backtest/intrabar_entry.py — checking the SAME 1h condition early,
every 5 minutes), these ask a genuinely different, additional question —
"does the most recent lower-timeframe price action itself show a
confirming pattern?" — evaluated exactly ONCE, at the same 1h-close
decision point the engine has always used. Execution timing is unchanged.

Two independent pieces, tested in isolation per the protocol (2A, then 2B,
never both at once until each is individually shown to help):

- lower_tf_rejection ("Opción A — rechazo"): the most recently closed
  lower-tf candle is itself a hammer-shaped rejection (reuses
  src.analysis.candles.is_hammer — same shape already used for the 1h
  candlestick confirmation, just applied one level down).
- lower_tf_structure_break ("Opción B — ruptura de estructura"): the most
  recently closed lower-tf candle's close breaks above the highest high of
  the `lookback` lower-tf candles before it (same Donchian-style logic as
  setups/breakout.py, just applied to the finer window).
"""
from __future__ import annotations

from src.analysis.candles import is_hammer
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("lower_tf_rejection")
def lower_tf_rejection(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    if ctx.lower_tf_window is None or ctx.lower_tf_window.empty:
        return None

    lower_wick_min_ratio = params.get("lower_wick_min_ratio", 2.0)
    upper_wick_max_ratio = params.get("upper_wick_max_ratio", 0.3)

    last = ctx.lower_tf_window.iloc[-1]
    if not is_hammer(last, lower_wick_min_ratio=lower_wick_min_ratio, upper_wick_max_ratio=upper_wick_max_ratio):
        return None
    return ConfirmationResult(name="lower_tf_rejection", extras={"lower_tf_rejection": True})


@CONFIRMATIONS.register("lower_tf_structure_break")
def lower_tf_structure_break(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    if ctx.lower_tf_window is None or ctx.lower_tf_window.empty:
        return None

    lookback = params.get("lookback", 6)
    window = ctx.lower_tf_window
    if len(window) < lookback + 1:
        return None

    # Highest high of the `lookback` lower-tf candles strictly before the
    # last closed one — that candle's own high never counts toward its own
    # breakout level (same discipline as setups/breakout.py).
    prior_high = window["high"].iloc[-(lookback + 1) : -1].max()
    last_close = window["close"].iloc[-1]

    if last_close <= prior_high:
        return None
    return ConfirmationResult(name="lower_tf_structure_break", extras={"lower_tf_break_level": prior_high})
