"""Confirmation pieces: candle geometry at the entry candle — continuous
proportions (marubozu conviction) and range-vs-recent-history (NR7
compression), see src.analysis.candle_geometry for the underlying
measures. Distinct from confirmations/candlestick.py's NAMED patterns
(hammer/engulfing/doji): these are simpler, purely geometric rules.

Rescued from a colleague's exploratory branch (probabilities-to-into-trade,
David A) and reimplemented as catalog pieces. Not wired into any strategy
YAML yet — available for an ablation test (see
scripts/run_rescued_ideas_ablation.py) alongside the existing catalog,
same "test before adopting" discipline as everything else here.
"""
from __future__ import annotations

from src.analysis.candle_geometry import is_marubozu, is_narrow_range
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("marubozu")
def marubozu(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    """Passes when the entry candle is a bullish marubozu (body >=
    `body_ratio_min`, default 0.9, of its own range, and closes above its
    open) — strong directional conviction, no doubt in the market on that
    candle. Bullish-only for now, mirrors confirmations/candlestick.py's
    own convention (the catalog is long/uptrend-only so far)."""
    body_ratio_min = params.get("body_ratio_min", 0.9)
    df = ctx.price_window
    if df.empty:
        return None
    row = df.iloc[[-1]]
    if not bool(is_marubozu(row, body_ratio_min=body_ratio_min).iloc[0]):
        return None
    if not (row["close"].iloc[0] > row["open"].iloc[0]):
        return None
    return ConfirmationResult(name="marubozu", extras={})


@CONFIRMATIONS.register("narrow_range")
def narrow_range(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    """Passes when the entry candle is a Narrow-Range-N (Crabel, 1990):
    its range is the tightest of the last `lookback` candles (default 7,
    the classic NR7) — compression right before the entry, which tends to
    precede a strong breakout."""
    lookback = params.get("lookback", 7)
    df = ctx.price_window
    if len(df) < lookback:
        return None
    flags = is_narrow_range(df, lookback=lookback)
    if not bool(flags.iloc[-1]):
        return None
    return ConfirmationResult(name="narrow_range", extras={})
