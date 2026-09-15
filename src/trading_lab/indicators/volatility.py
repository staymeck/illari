"""Volatility: Average True Range (ATR), used by more than one strategy to
define stop loss / take profit in units comparable to recent market noise
(instead of a fixed price value)."""

from __future__ import annotations

import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def volatility_regime_ratio(atr_series: pd.Series, lookback: int = 20) -> pd.Series:
    """Current ATR relative to its own recent moving average: > 1 means
    volatility is expanding right now (the market is "breathing" more than
    usual), < 1 means it's contracted. This is the market-state signal used
    to make the stop distance and the position size dynamic, instead of
    fixed — see `regime_multiplier`."""
    atr_sma = atr_series.rolling(window=lookback, min_periods=1).mean()
    ratio = atr_series / atr_sma
    return ratio.replace([float("inf"), float("-inf")], pd.NA)


def regime_multiplier(ratio: pd.Series, min_mult: float = 0.75, max_mult: float = 2.0) -> pd.Series:
    """Clips the raw volatility ratio (`volatility_regime_ratio`) to a sane
    range and fills missing values (not enough history yet) with 1.0 (no
    adjustment). Used two ways:
      - Stop distance: multiplied directly -> wider stop when volatility is
        expanding (so normal noise doesn't trigger it), tighter when it's
        contracted.
      - Position size: multiplied by its INVERSE (1/ratio) -> less capital
        risked per trade when volatility (and thus uncertainty) is
        elevated, more when the market is calm. Deliberately NOT based on
        the strategy's own recent win/loss streak (see report discussion):
        streak-based sizing bets on an illusion (no evidence a hot streak
        predicts the next trade), while volatility-based sizing responds to
        something actually measurable about current market conditions.
    """
    return ratio.clip(lower=min_mult, upper=max_mult).fillna(1.0)


def regime_multiplier_scalar(ratio: float | None, min_mult: float = 0.75, max_mult: float = 2.0) -> float:
    """Same as `regime_multiplier`, for a single row's value (used inside
    signal-generation loops, which evaluate one candle at a time) — `ratio`
    may be `None`/`NaN` (not enough history yet), which returns 1.0 (no
    adjustment)."""
    if ratio is None or pd.isna(ratio):
        return 1.0
    return min(max(ratio, min_mult), max_mult)
