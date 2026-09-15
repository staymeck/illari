import pandas as pd
import pytest

from trading_lab.indicators.volatility import (
    atr,
    regime_multiplier,
    regime_multiplier_scalar,
    volatility_regime_ratio,
)


def test_atr_uses_true_range_including_gaps():
    df = pd.DataFrame(
        {
            "high": [110.0, 115.0],
            "low": [90.0, 108.0],
            "close": [100.0, 112.0],
        }
    )
    result = atr(df, period=2)
    # Row 0: no prior close -> true range = high-low = 20.
    assert result.iloc[0] == pytest.approx(20.0)
    # Row 1: true range = max(115-108=7, |115-100|=15, |108-100|=8) = 15.
    # SMA(period=2, min_periods=1) of [20, 15] = 17.5.
    assert result.iloc[1] == pytest.approx(17.5)


def test_volatility_regime_ratio_above_one_when_expanding():
    # Flat ATR of 10 for a while, then a jump to 20 -> ratio > 1 on the jump.
    atr_series = pd.Series([10.0] * 5 + [20.0])
    ratio = volatility_regime_ratio(atr_series, lookback=5)
    assert ratio.iloc[-1] > 1.0


def test_volatility_regime_ratio_below_one_when_contracting():
    atr_series = pd.Series([20.0] * 5 + [5.0])
    ratio = volatility_regime_ratio(atr_series, lookback=5)
    assert ratio.iloc[-1] < 1.0


def test_regime_multiplier_clips_to_bounds():
    ratio = pd.Series([0.1, 1.0, 10.0, float("nan")])
    result = regime_multiplier(ratio, min_mult=0.75, max_mult=2.0)
    assert result.iloc[0] == pytest.approx(0.75)  # clipped up from 0.1
    assert result.iloc[1] == pytest.approx(1.0)    # unchanged, inside bounds
    assert result.iloc[2] == pytest.approx(2.0)    # clipped down from 10.0
    assert result.iloc[3] == pytest.approx(1.0)    # NaN -> no adjustment


def test_regime_multiplier_scalar_matches_series_version():
    assert regime_multiplier_scalar(0.1, 0.75, 2.0) == pytest.approx(0.75)
    assert regime_multiplier_scalar(1.3, 0.75, 2.0) == pytest.approx(1.3)
    assert regime_multiplier_scalar(10.0, 0.75, 2.0) == pytest.approx(2.0)
    assert regime_multiplier_scalar(None, 0.75, 2.0) == pytest.approx(1.0)
    assert regime_multiplier_scalar(float("nan"), 0.75, 2.0) == pytest.approx(1.0)
