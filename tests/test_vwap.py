"""Tests on synthetic data for src/analysis/vwap.py."""
import pandas as pd

from src.analysis.vwap import rolling_vwap, vwap_bias


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_rolling_vwap_known_value():
    # 2 bars: typical price (h+l+c)/3 = (12+8+10)/3=10 and (22+18+20)/3=20,
    # volumes 1 and 3 -> VWAP = (10*1 + 20*3)/(1+3) = 70/4 = 17.5
    df = _df(
        [
            {"high": 12.0, "low": 8.0, "close": 10.0, "volume": 1.0},
            {"high": 22.0, "low": 18.0, "close": 20.0, "volume": 3.0},
        ]
    )

    result = rolling_vwap(df, window=2)

    assert result.iloc[-1] == 17.5


def test_vwap_bias_positive_when_price_above_vwap():
    df = _df(
        [
            {"high": 12.0, "low": 8.0, "close": 10.0, "volume": 1.0},
            {"high": 22.0, "low": 18.0, "close": 25.0, "volume": 3.0},  # closes above its own typical price
        ]
    )

    bias = vwap_bias(df, window=2)

    assert bias > 0


def test_vwap_bias_zero_when_not_enough_data():
    df = _df([{"high": 12.0, "low": 8.0, "close": 10.0, "volume": 1.0}])

    assert vwap_bias(df, window=5) == 0.0
