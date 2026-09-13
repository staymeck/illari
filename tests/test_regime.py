"""Tests on synthetic data for src/analysis/regime.py."""
import numpy as np
import pandas as pd

from src.analysis.regime import trend_strength_regime, volatility_percentile


def test_volatility_percentile_high_when_current_atr_is_the_largest():
    # 100 bars of constant range 1, then a sudden much wider range at the
    # very end -> current ATR should sit at (or near) the 100th percentile
    # of its own trailing history.
    n = 130
    high = np.concatenate([np.full(n - 1, 101.0), [120.0]])
    low = np.concatenate([np.full(n - 1, 100.0), [90.0]])
    close = np.concatenate([np.full(n - 1, 100.5), [105.0]])
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    pct = volatility_percentile(df, atr_period=14, lookback=100)

    assert pct > 90


def test_volatility_percentile_low_when_current_atr_is_the_smallest():
    n = 130
    high = np.concatenate([np.full(n - 1, 110.0), [100.5]])
    low = np.concatenate([np.full(n - 1, 90.0), [100.0]])
    close = np.concatenate([np.full(n - 1, 100.0), [100.2]])
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    pct = volatility_percentile(df, atr_period=14, lookback=100)

    assert pct < 10


def test_volatility_percentile_nan_before_warmup():
    df = pd.DataFrame({"high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.0, 101.0]})

    assert pd.isna(volatility_percentile(df, atr_period=14, lookback=100))


def test_trend_strength_regime_trending_for_a_strong_move():
    values = np.arange(1, 61)
    df = pd.DataFrame({"high": values + 1, "low": values - 1, "close": values})

    assert trend_strength_regime(df, adx_period=14, trending_threshold=25) == "trending"


def test_trend_strength_regime_choppy_for_a_sideways_range():
    n = 60
    close = 100 + np.sin(np.arange(n))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})

    assert trend_strength_regime(df, adx_period=14, trending_threshold=25) == "choppy"


def test_trend_strength_regime_unknown_before_warmup():
    df = pd.DataFrame({"high": [2.0, 3.0], "low": [1.0, 2.0], "close": [1.5, 2.5]})

    assert trend_strength_regime(df) == "unknown"
