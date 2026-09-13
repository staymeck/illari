"""Average Directional Index (ADX) — a trend-*strength* indicator, distinct
from trend *direction* (dow_trend/ma_trend already cover direction). A weak,
choppy uptrend and a strong, clean uptrend both classify as "uptrend" today;
only one has real conviction behind it.

Reference: docs/PLAN.md, "Config-driven strategy catalog"; the ADX section
of resources/12_claves_analisis_tecnico.pdf.
"""
from __future__ import annotations

import pandas as pd


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX. Conventionally: <20 weak/no trend, >25 a trend with
    real strength, >40 a strong trend."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    smoothed_tr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
