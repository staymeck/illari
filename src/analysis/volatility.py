"""Volatility indicators (ATR, Bollinger Bands, z-score) for the strategy
catalog — used both to gauge a volatility regime and as mean-reversion
setup/risk anchors.

Reference: docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.moving_average import sma


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range: the smoothed average of the true range
    (the largest of high-low, |high-prev_close|, |low-prev_close|)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: a moving-average midline plus/minus `num_std`
    rolling standard deviations."""
    mid = sma(series, window)
    std = series.rolling(window).std()
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std})


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """How many rolling standard deviations `series` currently sits from its
    own rolling mean — a normalized mean-reversion signal."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std
