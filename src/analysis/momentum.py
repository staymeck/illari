"""Momentum indicators (RSI, MACD) for the "confirmation" layer of the
strategy catalog.

Reference: docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.moving_average import ema


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index. 100 means only gains have occurred
    in the smoothing window, 0 means only losses."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), its signal line (EMA of the MACD
    line), and the histogram (MACD - signal)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})
