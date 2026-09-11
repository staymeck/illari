"""Candlestick patterns used as confirmation (never as a standalone signal).

All functions are vectorized over an OHLC DataFrame and return a boolean
Series. The empirical evidence for candlestick patterns alone is weak —
they're computed here to combine with structure + Fibonacci + volume in
the confluence strategy, not to be traded by themselves.
"""

from __future__ import annotations

import pandas as pd


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    # np.nan (not pd.NA): comparing against pd.NA with pandas' numexpr
    # backend breaks with "boolean value of NA is ambiguous"; np.nan is a
    # native float and NaN comparisons simply give False, no ambiguity.
    return (df["high"] - df["low"]).replace(0, float("nan"))


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    prev_bearish = prev_close < prev_open
    curr_bullish = df["close"] > df["open"]
    engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    return (prev_bearish & curr_bullish & engulfs).fillna(False)


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    prev_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return (prev_bullish & curr_bearish & engulfs).fillna(False)


def is_hammer(df: pd.DataFrame, body_ratio_max: float = 0.35, lower_wick_ratio_min: float = 2.0) -> pd.Series:
    """Hammer: small body in the upper part of the range, long lower wick
    (rejection of lower prices -> buying pressure)."""
    rng = _range(df)
    body = _body(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    small_body = body <= body_ratio_max * rng
    long_lower = lower >= lower_wick_ratio_min * body.replace(0, 0.0001)
    small_upper = upper <= body.clip(lower=0.0001) * 1.0
    return (small_body & long_lower & small_upper).fillna(False)


def is_shooting_star(df: pd.DataFrame, body_ratio_max: float = 0.35, upper_wick_ratio_min: float = 2.0) -> pd.Series:
    """Shooting star: small body in the lower part of the range, long upper
    wick (rejection of higher prices -> selling pressure)."""
    rng = _range(df)
    body = _body(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    small_body = body <= body_ratio_max * rng
    long_upper = upper >= upper_wick_ratio_min * body.replace(0, 0.0001)
    small_lower = lower <= body.clip(lower=0.0001) * 1.0
    return (small_body & long_upper & small_lower).fillna(False)


def is_doji(df: pd.DataFrame, body_ratio_max: float = 0.1) -> pd.Series:
    rng = _range(df)
    body = _body(df)
    return (body <= body_ratio_max * rng).fillna(False)


def is_bullish_pin_bar(df: pd.DataFrame) -> pd.Series:
    """Bullish pin bar: a hammer variant with somewhat looser requirements,
    used as the entry trigger in the confluence strategy."""
    return is_hammer(df, body_ratio_max=0.4, lower_wick_ratio_min=1.8)


def is_bearish_pin_bar(df: pd.DataFrame) -> pd.Series:
    return is_shooting_star(df, body_ratio_max=0.4, upper_wick_ratio_min=1.8)


def bullish_confirmation(df: pd.DataFrame) -> pd.Series:
    """True if the candle is any of the bullish confirmation patterns."""
    return is_bullish_engulfing(df) | is_hammer(df) | is_bullish_pin_bar(df)


def bearish_confirmation(df: pd.DataFrame) -> pd.Series:
    return is_bearish_engulfing(df) | is_shooting_star(df) | is_bearish_pin_bar(df)
