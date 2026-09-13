"""Shared types passed between the strategy catalog's pieces (context, setup,
confirmation, risk) and the generic engine that wires them together.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EvalContext:
    """Everything a catalog piece might need to evaluate the current bar.

    `price_window` is the trailing OHLCV window up to and including the
    current bar (no look-ahead). `marked_window` is the precomputed swing
    pivots confirmed as of the current bar (see src/backtest/engine.py) —
    only structure-based pieces (dow_trend, support_touch, fibonacci) use it;
    others (moving averages, RSI, VWAP, ...) just ignore it.

    `higher_tf_window` (optional, None unless the engine was given a second,
    higher timeframe's data — see src/backtest/engine.run_backtest's
    `higher_tf_df` parameter) is the trailing window of *already-closed*
    higher-timeframe candles as of the current bar — used by
    confirmations/higher_tf_trend.py (Kaufman, Trading Systems and Methods,
    ch. 19 "Multiple Time Frames"). Pieces that don't need it just ignore it,
    same as marked_window.
    """

    price_window: pd.DataFrame
    marked_window: pd.DataFrame
    higher_tf_window: pd.DataFrame | None = None


@dataclass(frozen=True)
class SetupResult:
    """What a setup piece found: a reference price the risk pieces anchor
    stops to (e.g. a support level, a breakout level, a Bollinger band),
    plus whatever extra fields are worth recording on the trade for
    reporting."""

    reference_level: float
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConfirmationResult:
    """A passed confirmation: its piece name plus any extras worth recording
    on the trade (e.g. which Fibonacci ratio matched, which candle
    pattern)."""

    name: str
    extras: dict = field(default_factory=dict)
