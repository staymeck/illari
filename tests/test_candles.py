import pandas as pd

from trading_lab.indicators.candles import (
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_hammer,
    is_shooting_star,
)


def test_bullish_engulfing_detected():
    df = pd.DataFrame(
        {
            "open": [110, 95],
            "high": [112, 116],
            "low": [94, 94],
            "close": [95, 115],  # previous bearish candle (110->95), current bullish candle engulfs it
        }
    )
    result = is_bullish_engulfing(df)
    assert result.iloc[0] == False  # noqa: E712 (no previous candle)
    assert result.iloc[1] == True  # noqa: E712


def test_bearish_engulfing_detected():
    df = pd.DataFrame(
        {
            "open": [95, 110],
            "high": [116, 112],
            "low": [94, 89],
            "close": [110, 90],  # previous bullish candle (95->110), current bearish candle engulfs it
        }
    )
    result = is_bearish_engulfing(df)
    assert result.iloc[1] == True  # noqa: E712


def test_no_engulfing_on_small_candle():
    df = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [105, 103],
            "low": [95, 100],
            "close": [95, 102],  # small candle, doesn't engulf the previous one
        }
    )
    assert is_bullish_engulfing(df).iloc[1] == False  # noqa: E712


def test_hammer_detected():
    # Small body at the top of the range, long lower wick, minimal upper wick.
    df = pd.DataFrame({"open": [100], "high": [101], "low": [80], "close": [101 - 0.5]})
    assert is_hammer(df).iloc[0] == True  # noqa: E712


def test_shooting_star_detected():
    # Small body at the bottom of the range, long upper wick, minimal lower wick.
    df = pd.DataFrame({"open": [100], "high": [120], "low": [99.8], "close": [100.5]})
    assert is_shooting_star(df).iloc[0] == True  # noqa: E712


def test_doji_detected():
    df = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [100.5]})
    assert is_doji(df).iloc[0] == True  # noqa: E712


def test_not_doji_on_large_body():
    df = pd.DataFrame({"open": [100], "high": [130], "low": [90], "close": [125]})
    assert is_doji(df).iloc[0] == False  # noqa: E712
