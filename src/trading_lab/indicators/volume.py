"""Volume as a proxy for the imbalance between buyers and sellers.

Public spot OHLCV data doesn't carry real order flow (tick-by-tick
buy/sell delta). As a reasonable, widely used approximation when that
granularity isn't available, we use the position of the close within the
candle's range weighted by volume: a close near the high with high volume
suggests dominant buying pressure, near the low suggests selling pressure.
"""

from __future__ import annotations

import pandas as pd


def volume_sma(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["volume"].rolling(window=lookback, min_periods=1).mean()


def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Each candle's volume relative to its moving average (>1 = above average)."""
    avg = volume_sma(df, lookback)
    return df["volume"] / avg.replace(0, float("nan"))


def is_volume_sufficient(df: pd.DataFrame, lookback: int = 20, min_ratio: float = 0.8) -> pd.Series:
    """True if the candle's volume isn't anomalously low relative to the average."""
    return (volume_ratio(df, lookback) >= min_ratio).fillna(False)


def buy_sell_pressure_proxy(df: pd.DataFrame) -> pd.Series:
    """Buy/sell delta proxy per candle, in [-1, 1].

    Close to +1: close near the high (buyer dominance).
    Close to -1: close near the low (seller dominance).
    It's weighted by volume so that volume spikes carry more weight in the
    context of a zone, even though the per-candle value is already normalized.
    """
    rng = (df["high"] - df["low"]).replace(0, float("nan"))
    close_position = (df["close"] - df["low"]) / rng  # 0 (at the low) .. 1 (at the high)
    return (close_position * 2 - 1).fillna(0.0)


def volume_weighted_pressure(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Buy/sell pressure weighted by recent relative volume."""
    pressure = buy_sell_pressure_proxy(df)
    weight = volume_ratio(df, lookback).fillna(1.0)
    return pressure * weight
