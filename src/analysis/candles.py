"""Candlestick pattern detection.

Per docs/PLAN.md ("Candlestick patterns... only considered combined with
structural context, never as a signal on their own"): these functions only
recognize the raw geometric pattern. Deciding whether a pattern matters — e.g.
requiring a hammer to appear at a support/Fibonacci confluence zone — is the
caller's responsibility (see src/backtest/engine.py).
"""
from __future__ import annotations

import pandas as pd


def body(row: pd.Series) -> float:
    return abs(row["close"] - row["open"])


def candle_range(row: pd.Series) -> float:
    return row["high"] - row["low"]


def upper_wick(row: pd.Series) -> float:
    return row["high"] - max(row["close"], row["open"])


def lower_wick(row: pd.Series) -> float:
    return min(row["close"], row["open"]) - row["low"]


def is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def is_doji(row: pd.Series, body_to_range_max: float = 0.1) -> bool:
    """Body is a tiny fraction of the candle's total range — an indecision candle."""
    rng = candle_range(row)
    if rng == 0:
        return True
    return body(row) / rng <= body_to_range_max


def is_hammer(row: pd.Series, lower_wick_min_ratio: float = 2.0, upper_wick_max_ratio: float = 0.3) -> bool:
    """Small body near the top of the range, a lower wick at least
    `lower_wick_min_ratio` times the body, and little to no upper wick — a
    bullish-reversal candle when it appears after a decline (context is the
    caller's responsibility)."""
    b = body(row)
    if b == 0:
        return False
    return lower_wick(row) >= lower_wick_min_ratio * b and upper_wick(row) <= upper_wick_max_ratio * b


def is_shooting_star(row: pd.Series, upper_wick_min_ratio: float = 2.0, lower_wick_max_ratio: float = 0.3) -> bool:
    """Mirror image of the hammer: small body near the bottom of the range and
    a long upper wick — a bearish-reversal candle when it appears after an advance."""
    b = body(row)
    if b == 0:
        return False
    return upper_wick(row) >= upper_wick_min_ratio * b and lower_wick(row) <= lower_wick_max_ratio * b


def is_bullish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    """A bearish candle followed by a bullish candle whose body fully engulfs
    the previous candle's body."""
    if not (is_bearish(prev_row) and is_bullish(row)):
        return False
    return row["open"] <= prev_row["close"] and row["close"] >= prev_row["open"]


def is_bearish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    """A bullish candle followed by a bearish candle whose body fully engulfs
    the previous candle's body."""
    if not (is_bullish(prev_row) and is_bearish(row)):
        return False
    return row["open"] >= prev_row["close"] and row["close"] <= prev_row["open"]


def bullish_reversal_pattern(df: pd.DataFrame, idx: int) -> str | None:
    """Checks candle `idx` (and `idx - 1` where needed) for a known bullish
    reversal pattern. Returns the pattern name, or None."""
    row = df.iloc[idx]
    if is_hammer(row):
        return "hammer"
    if idx > 0 and is_bullish_engulfing(df.iloc[idx - 1], row):
        return "bullish_engulfing"
    return None


def bearish_reversal_pattern(df: pd.DataFrame, idx: int) -> str | None:
    """Checks candle `idx` (and `idx - 1` where needed) for a known bearish
    reversal pattern. Returns the pattern name, or None."""
    row = df.iloc[idx]
    if is_shooting_star(row):
        return "shooting_star"
    if idx > 0 and is_bearish_engulfing(df.iloc[idx - 1], row):
        return "bearish_engulfing"
    return None
