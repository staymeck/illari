"""Individual candle geometry: instead of named patterns (engulfing/
hammer/shooting star, see `candles.py`), here we measure candle anatomy as
continuous proportions (body/wick as % of range) and the current candle's
range compared to its own recent history — simpler mathematical rules,
documented in quantitative price action:

  - Marubozu: body ≈ 100% of the range (almost no wicks) -> strong
    directional conviction, no doubt in the market on that candle.
  - NR7 (Narrow Range 7) — Toby Crabel, "Day Trading with Short Term Price
    Patterns" (1990): the current candle's range is the narrowest of the
    last N -> tends to precede a strong breakout (compression before the
    move).
  - WR7 (Wide Range 7): the opposite — widest range of the last N ->
    momentum already under way.
  - Inside bar: the current range is fully contained within the previous
    candle's range -> consolidation.

All functions are vectorized and lookahead-free: they only look backward
(the current candle and its own recent history).
"""

from __future__ import annotations

import pandas as pd


def candle_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, float("nan"))


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """Body as a fraction of the total range (0 = perfect doji, 1 = marubozu)."""
    body = (df["close"] - df["open"]).abs()
    return (body / candle_range(df)).fillna(0.0)


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper wick as a fraction of the total range."""
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    return (upper_wick / candle_range(df)).fillna(0.0)


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Lower wick as a fraction of the total range."""
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    return (lower_wick / candle_range(df)).fillna(0.0)


def is_marubozu(df: pd.DataFrame, body_ratio_min: float = 0.9) -> pd.Series:
    """Body >= `body_ratio_min` of the range — almost no wicks, strong conviction."""
    return (body_ratio(df) >= body_ratio_min).fillna(False)


def is_narrow_range(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """Generalized NR7: candle `i`'s range is the minimum among the
    `lookback` most recent candles (including itself). Only looks
    backward — the `rolling` window on row `i` only uses rows `<= i`."""
    rng = candle_range(df).fillna(0.0)
    rolling_min = rng.rolling(window=lookback, min_periods=lookback).min()
    return (rng == rolling_min).fillna(False)


def is_wide_range(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """Generalized WR7: candle `i`'s range is the maximum among the
    `lookback` most recent candles (including itself)."""
    rng = candle_range(df).fillna(0.0)
    rolling_max = rng.rolling(window=lookback, min_periods=lookback).max()
    return (rng == rolling_max).fillna(False)


def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    """The current candle's range is fully contained within the previous
    candle's range (consolidation)."""
    prev_high, prev_low = df["high"].shift(1), df["low"].shift(1)
    return ((df["high"] <= prev_high) & (df["low"] >= prev_low)).fillna(False)
