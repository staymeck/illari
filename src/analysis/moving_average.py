"""Moving averages and MA-based trend classification — an alternative to
Dow-theory swing analysis (structure.classify_trend) for the "context" layer
of the strategy catalog.

Reference: docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _moving_average(series: pd.Series, window: int, method: str) -> pd.Series:
    if method == "sma":
        return sma(series, window)
    if method == "ema":
        return ema(series, window)
    raise ValueError(f"Unknown moving average method: {method!r} (expected 'sma' or 'ema')")


def ma_cross_trend(df: pd.DataFrame, fast: int = 20, slow: int = 50, method: str = "ema") -> str:
    """Classifies the trend by comparing a fast and a slow moving average of
    the close price: "uptrend" when fast > slow, "downtrend" when fast <
    slow, "sideways" when there isn't enough data yet for both averages."""
    if len(df) < slow:
        return "sideways"

    fast_ma = _moving_average(df["close"], fast, method)
    slow_ma = _moving_average(df["close"], slow, method)
    fast_last = fast_ma.iloc[-1]
    slow_last = slow_ma.iloc[-1]

    if pd.isna(fast_last) or pd.isna(slow_last):
        return "sideways"
    if fast_last > slow_last:
        return "uptrend"
    if fast_last < slow_last:
        return "downtrend"
    return "sideways"
