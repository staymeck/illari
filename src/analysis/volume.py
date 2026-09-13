"""Volume-based buying/selling pressure analysis.

Standard OHLCV candles don't split volume into buy/sell size, so this module
uses two objective proxies instead of guessing:

1. Relative volume: a candle's volume versus its trailing rolling average —
   flags unusually strong (or weak) participation at a given candle.
2. Directional volume bias: over a trailing window, whether volume has been
   concentrated more on bullish or bearish candles.

Reference: docs/PLAN.md, "Technical analysis engine components", item 4.
"""
from __future__ import annotations

import pandas as pd


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Ratio of each candle's volume to the trailing rolling average volume,
    excluding the candle itself (via `shift(1)`) to avoid look-ahead within
    the average."""
    rolling_avg = df["volume"].shift(1).rolling(window).mean()
    return df["volume"] / rolling_avg


def is_volume_spike(df: pd.DataFrame, idx: int, window: int = 20, threshold: float = 1.5) -> bool:
    """Whether candle `idx` has meaningfully above-average volume, computed
    using only data up to `idx` (no future)."""
    rel = relative_volume(df.iloc[: idx + 1], window=window)
    value = rel.iloc[-1]
    return bool(pd.notna(value) and value >= threshold)


def directional_volume_bias(df: pd.DataFrame, window: int = 20) -> float:
    """Share of trailing volume tied to bullish candles minus the share tied
    to bearish candles, over the last `window` candles of `df`. Ranges from
    -1 (all volume on down candles) to +1 (all volume on up candles)."""
    recent = df.tail(window)
    total = recent["volume"].sum()
    if recent.empty or total == 0:
        return 0.0
    bullish_volume = recent.loc[recent["close"] > recent["open"], "volume"].sum()
    bearish_volume = recent.loc[recent["close"] < recent["open"], "volume"].sum()
    return float((bullish_volume - bearish_volume) / total)


def confirms_buyer_pressure(
    df: pd.DataFrame,
    idx: int,
    window: int = 20,
    spike_threshold: float = 1.2,
    min_bias: float = 0.1,
) -> bool:
    """Confirmation filter: the candle at `idx` shows above-average volume AND
    the trailing window is net buyer-dominant — so a structure+Fibonacci+candle
    setup isn't taken on thin participation. Uses only data up to `idx`."""
    if not is_volume_spike(df, idx, window=window, threshold=spike_threshold):
        return False
    bias = directional_volume_bias(df.iloc[: idx + 1], window=window)
    return bias >= min_bias
