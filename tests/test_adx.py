"""Tests on synthetic data for src/analysis/adx.py.

ADX's exact formula is a chain of Wilder smoothings that isn't practical to
hand-verify to the last decimal (same reasoning as the RSI/MACD tests) — so,
same as those, these check the property that actually matters: a strong,
clean trend scores much higher than a choppy, directionless one.
"""
import numpy as np
import pandas as pd

from src.analysis.adx import adx


def test_adx_high_for_a_strong_clean_trend():
    n = 60
    values = np.arange(1, n + 1)
    df = pd.DataFrame({"high": values + 1, "low": values - 1, "close": values})

    result = adx(df, period=14)

    assert result.iloc[-1] > 40  # conventionally a "strong trend" reading


def test_adx_low_for_a_choppy_range():
    n = 60
    close = 100 + np.sin(np.arange(n))  # oscillates, no sustained direction
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})

    result = adx(df, period=14)

    assert result.iloc[-1] < 20  # conventionally a "weak/no trend" reading


def test_adx_nan_before_period_warms_up():
    df = pd.DataFrame({"high": [2.0, 3.0], "low": [1.0, 2.0], "close": [1.5, 2.5]})

    result = adx(df, period=14)

    assert result.isna().all()
