import pandas as pd
import pytest

from trading_lab.indicators.candle_geometry import (
    body_ratio,
    is_inside_bar,
    is_marubozu,
    is_narrow_range,
    is_wide_range,
    lower_wick_ratio,
    upper_wick_ratio,
)


def _one_candle(open_, high, low, close) -> pd.DataFrame:
    return pd.DataFrame({"open": [open_], "high": [high], "low": [low], "close": [close]})


def test_body_ratio_known_candle():
    df = _one_candle(100, 112, 98, 110)  # rango=14, cuerpo=10
    assert body_ratio(df).iloc[0] == pytest.approx(10 / 14)


def test_upper_wick_ratio_known_candle():
    df = _one_candle(100, 112, 98, 110)  # mecha superior = 112-110 = 2
    assert upper_wick_ratio(df).iloc[0] == pytest.approx(2 / 14)


def test_lower_wick_ratio_known_candle():
    df = _one_candle(100, 112, 98, 110)  # mecha inferior = 100-98 = 2
    assert lower_wick_ratio(df).iloc[0] == pytest.approx(2 / 14)


def test_is_marubozu_true_without_wicks():
    df = _one_candle(100, 110, 100, 110)  # sin mechas, cuerpo = 100% del rango
    assert is_marubozu(df).iloc[0] == True  # noqa: E712


def test_is_marubozu_false_with_large_wicks():
    df = _one_candle(100, 115, 95, 105)  # cuerpo = 5, rango = 20 -> ratio 0.25
    assert is_marubozu(df).iloc[0] == False  # noqa: E712


def test_is_narrow_range_detects_smallest_window():
    ranges = [10, 8, 6, 9, 3]
    df = pd.DataFrame({"high": ranges, "low": [0] * 5, "open": [0] * 5, "close": [0] * 5})
    result = is_narrow_range(df, lookback=3)

    assert list(result.iloc[:2]) == [False, False]  # ventana incompleta
    assert result.iloc[2] == True   # min([10,8,6])=6, rango actual=6 -> noqa: E712
    assert result.iloc[3] == False  # min([8,6,9])=6, rango actual=9
    assert result.iloc[4] == True   # min([6,9,3])=3, rango actual=3 -> noqa: E712


def test_is_wide_range_detects_largest_window():
    ranges = [10, 8, 6, 9, 3]
    df = pd.DataFrame({"high": ranges, "low": [0] * 5, "open": [0] * 5, "close": [0] * 5})
    result = is_wide_range(df, lookback=3)

    assert list(result.iloc[:2]) == [False, False]
    assert result.iloc[2] == False  # max([10,8,6])=10, rango actual=6
    assert result.iloc[3] == True   # max([8,6,9])=9, rango actual=9 -> noqa: E712
    assert result.iloc[4] == False  # max([6,9,3])=9, rango actual=3


def test_is_inside_bar_true_when_fully_contained():
    df = pd.DataFrame({"high": [110, 108], "low": [95, 100], "open": [100, 103], "close": [105, 104]})
    result = is_inside_bar(df)
    assert result.iloc[0] == False  # noqa: E712 (sin vela previa)
    assert result.iloc[1] == True   # noqa: E712 (108<=110 y 100>=95)


def test_is_inside_bar_false_when_range_extends_beyond():
    df = pd.DataFrame({"high": [110, 112], "low": [95, 100], "open": [100, 103], "close": [105, 104]})
    result = is_inside_bar(df)
    assert result.iloc[1] == False  # noqa: E712 (112 > 110)
