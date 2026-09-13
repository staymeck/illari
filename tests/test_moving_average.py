"""Tests on synthetic data for src/analysis/moving_average.py."""
import pandas as pd
import pytest

from src.analysis.moving_average import ema, ma_cross_trend, sma


def test_sma_known_value():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = sma(series, window=3)

    assert result.iloc[-1] == 4.0  # mean(3, 4, 5)
    assert pd.isna(result.iloc[0])  # not enough data yet


def test_ema_matches_hand_computed_recursion():
    series = pd.Series([1.0, 2.0, 3.0])
    span = 2
    alpha = 2 / (span + 1)

    result = ema(series, span=span)

    expected_0 = 1.0
    expected_1 = alpha * 2.0 + (1 - alpha) * expected_0
    expected_2 = alpha * 3.0 + (1 - alpha) * expected_1
    assert result.iloc[0] == expected_0
    assert result.iloc[1] == pytest.approx(expected_1)
    assert result.iloc[2] == pytest.approx(expected_2)


def test_ma_cross_trend_uptrend_when_fast_above_slow():
    # Clear, sustained rally: fast EMA pulls above slow EMA.
    closes = list(range(1, 101))  # 1..100, strong uptrend
    df = pd.DataFrame({"close": closes})

    assert ma_cross_trend(df, fast=5, slow=20, method="ema") == "uptrend"


def test_ma_cross_trend_downtrend_when_fast_below_slow():
    closes = list(range(100, 0, -1))  # strong downtrend
    df = pd.DataFrame({"close": closes})

    assert ma_cross_trend(df, fast=5, slow=20, method="ema") == "downtrend"


def test_ma_cross_trend_sideways_when_not_enough_data():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    assert ma_cross_trend(df, fast=5, slow=20) == "sideways"
