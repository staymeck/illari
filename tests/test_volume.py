import pandas as pd
import pytest

from trading_lab.indicators.volume import (
    buy_sell_pressure_proxy,
    is_volume_sufficient,
    volume_ratio,
    volume_sma,
    volume_weighted_pressure,
)


def _volume_df() -> pd.DataFrame:
    return pd.DataFrame({"volume": [10.0, 20.0, 30.0, 20.0, 20.0]})


def test_volume_sma_rolling_average():
    sma = volume_sma(_volume_df(), lookback=2)
    assert sma.iloc[0] == pytest.approx(10.0)   # no prior history, min_periods=1
    assert sma.iloc[1] == pytest.approx(15.0)   # (10+20)/2
    assert sma.iloc[2] == pytest.approx(25.0)   # (20+30)/2
    assert sma.iloc[4] == pytest.approx(20.0)   # (20+20)/2


def test_volume_ratio_relative_to_average():
    ratio = volume_ratio(_volume_df(), lookback=2)
    assert ratio.iloc[0] == pytest.approx(1.0)
    assert ratio.iloc[1] == pytest.approx(20.0 / 15.0)
    assert ratio.iloc[3] == pytest.approx(20.0 / 25.0)  # 0.8


def test_is_volume_sufficient_threshold():
    result = is_volume_sufficient(_volume_df(), lookback=2, min_ratio=0.9)
    assert list(result) == [True, True, True, False, True]


def test_buy_sell_pressure_proxy_bullish_and_bearish():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 110.0, 110.0],
            "low": [90.0, 90.0, 90.0],
            "close": [108.0, 92.0, 100.0],  # near the high / near the low / in the middle
        }
    )
    pressure = buy_sell_pressure_proxy(df)
    assert pressure.iloc[0] == pytest.approx(0.8)   # buyer dominance
    assert pressure.iloc[1] == pytest.approx(-0.8)  # seller dominance
    assert pressure.iloc[2] == pytest.approx(0.0)   # neutral


def test_buy_sell_pressure_proxy_handles_zero_range():
    df = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})
    pressure = buy_sell_pressure_proxy(df)
    assert pressure.iloc[0] == pytest.approx(0.0)  # no range -> neutral, shouldn't blow up


def test_volume_weighted_pressure_scales_by_relative_volume():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [110.0, 110.0],
            "low": [90.0, 90.0],
            "close": [108.0, 108.0],
            "volume": [10.0, 30.0],
        }
    )
    weighted = volume_weighted_pressure(df, lookback=2)
    # Row 0: no prior history -> ratio=1 -> same as the raw pressure (0.8).
    assert weighted.iloc[0] == pytest.approx(0.8)
    # Row 1: ratio = 30 / ((10+30)/2) = 1.5 -> amplified pressure.
    assert weighted.iloc[1] == pytest.approx(0.8 * 1.5)
