"""Tests on synthetic data for src/analysis/momentum.py."""
import pandas as pd

from src.analysis.momentum import macd, rsi


def test_rsi_is_100_for_a_pure_uptrend():
    # Every step is a gain, so average loss stays 0 -> RSI = 100.
    series = pd.Series(range(1, 30))

    result = rsi(series, period=14)

    assert result.iloc[-1] == 100.0


def test_rsi_is_0_for_a_pure_downtrend():
    series = pd.Series(range(30, 1, -1))

    result = rsi(series, period=14)

    assert result.iloc[-1] == 0.0


def test_rsi_is_nan_before_the_period_warms_up():
    series = pd.Series([1.0, 2.0, 3.0])

    result = rsi(series, period=14)

    assert result.isna().all()


def test_macd_is_zero_for_a_constant_series():
    # Fast EMA == slow EMA == signal line when price never moves.
    series = pd.Series([100.0] * 40)

    result = macd(series, fast=12, slow=26, signal=9)

    assert (result["macd"] == 0).all()
    assert (result["signal"] == 0).all()
    assert (result["histogram"] == 0).all()


def test_macd_histogram_equals_macd_minus_signal():
    series = pd.Series([100 + i + (i % 5) for i in range(60)])  # noisy uptrend

    result = macd(series)

    diff = (result["histogram"] - (result["macd"] - result["signal"])).abs()
    assert (diff < 1e-9).all()
