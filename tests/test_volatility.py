"""Tests on synthetic data for src/analysis/volatility.py."""
import pandas as pd

from src.analysis.volatility import atr, bollinger_bands, z_score


def test_atr_converges_to_constant_true_range():
    # high - low is always 2, and close never gaps beyond [low, high], so
    # the true range is always exactly 2 -> ATR settles at 2.
    df = pd.DataFrame(
        {
            "high": [102.0] * 30,
            "low": [100.0] * 30,
            "close": [101.0] * 30,
        }
    )

    result = atr(df, period=14)

    assert result.iloc[-1] == 2.0
    assert result.iloc[:13].isna().all()  # not warmed up yet


def test_bollinger_bands_constant_series_has_zero_width():
    series = pd.Series([50.0] * 25)

    bands = bollinger_bands(series, window=20, num_std=2.0)

    assert bands["mid"].iloc[-1] == 50.0
    assert bands["upper"].iloc[-1] == 50.0
    assert bands["lower"].iloc[-1] == 50.0


def test_bollinger_bands_known_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = 3.0
    std = series.std()  # pandas default ddof=1

    bands = bollinger_bands(series, window=5, num_std=2.0)

    assert bands["mid"].iloc[-1] == mean
    assert bands["upper"].iloc[-1] == mean + 2 * std
    assert bands["lower"].iloc[-1] == mean - 2 * std


def test_z_score_known_value():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = 3.0
    std = series.std()

    result = z_score(series, window=5)

    assert result.iloc[-1] == (5.0 - mean) / std
