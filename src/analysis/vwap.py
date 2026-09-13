"""Rolling Volume Weighted Average Price and price-vs-VWAP bias, for the
strategy catalog's "confirmation" layer.

Reference: docs/PLAN.md, "Config-driven strategy catalog". This is a
rolling VWAP (over the last `window` bars) rather than a session-anchored
one — simpler to compute consistently across timeframes, and the same
"how far is price from its recent volume-weighted average" idea day traders
use session VWAP for.
"""
from __future__ import annotations

import pandas as pd


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    price_times_volume = typical_price * df["volume"]
    return price_times_volume.rolling(window).sum() / df["volume"].rolling(window).sum()


def vwap_bias(df: pd.DataFrame, window: int = 20) -> float:
    """The last close's % distance from the rolling VWAP: positive means
    price is trading above VWAP (bullish bias), negative below. Returns 0.0
    when there isn't enough data yet."""
    vwap = rolling_vwap(df, window=window)
    last_vwap = vwap.iloc[-1]
    if pd.isna(last_vwap) or last_vwap == 0:
        return 0.0
    last_close = df["close"].iloc[-1]
    return float((last_close - last_vwap) / last_vwap * 100)
