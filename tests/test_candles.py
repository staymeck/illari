"""Tests on synthetic data for src/analysis/candles.py."""
import pandas as pd

from src.analysis.candles import (
    bearish_reversal_pattern,
    bullish_reversal_pattern,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_hammer,
    is_shooting_star,
)


def _row(open_, high, low, close) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def test_is_hammer_true_for_small_body_long_lower_wick():
    # body [9, 10], lower wick down to 5 (5x body), tiny upper wick.
    row = _row(open_=9, high=10.1, low=5, close=10)

    assert is_hammer(row)


def test_is_hammer_false_for_long_upper_wick():
    row = _row(open_=9, high=15, low=5, close=10)

    assert not is_hammer(row)


def test_is_shooting_star_true_for_small_body_long_upper_wick():
    row = _row(open_=10, high=15, low=9.9, close=9)

    assert is_shooting_star(row)


def test_is_doji_true_for_tiny_body():
    row = _row(open_=10.0, high=11.0, low=9.0, close=10.02)

    assert is_doji(row)


def test_is_doji_false_for_large_body():
    row = _row(open_=10.0, high=11.0, low=9.0, close=10.9)

    assert not is_doji(row)


def test_is_bullish_engulfing_true():
    prev_row = _row(open_=10, high=10.2, low=8.8, close=9)  # bearish
    row = _row(open_=8.9, high=11.5, low=8.8, close=11)  # bullish, engulfs

    assert is_bullish_engulfing(prev_row, row)


def test_is_bullish_engulfing_false_when_body_does_not_engulf():
    prev_row = _row(open_=10, high=10.2, low=8.8, close=9)  # bearish
    row = _row(open_=9.2, high=10.0, low=9.0, close=9.8)  # bullish but too small

    assert not is_bullish_engulfing(prev_row, row)


def test_is_bearish_engulfing_true():
    prev_row = _row(open_=9, high=10.2, low=8.8, close=10)  # bullish
    row = _row(open_=10.1, high=10.2, low=7.5, close=8)  # bearish, engulfs

    assert is_bearish_engulfing(prev_row, row)


def test_bullish_reversal_pattern_detects_hammer():
    df = pd.DataFrame(
        [
            {"open": 12, "high": 12.1, "low": 10, "close": 11},
            {"open": 9, "high": 10.1, "low": 5, "close": 10},  # hammer
        ]
    )

    assert bullish_reversal_pattern(df, idx=1) == "hammer"


def test_bullish_reversal_pattern_detects_engulfing_when_not_a_hammer():
    df = pd.DataFrame(
        [
            {"open": 10, "high": 10.2, "low": 8.8, "close": 9},  # bearish
            {"open": 8.9, "high": 11.5, "low": 8.8, "close": 11},  # bullish engulfing
        ]
    )

    assert bullish_reversal_pattern(df, idx=1) == "bullish_engulfing"


def test_bullish_reversal_pattern_none_for_plain_candle():
    df = pd.DataFrame([{"open": 10, "high": 10.5, "low": 9.5, "close": 10.2}])

    assert bullish_reversal_pattern(df, idx=0) is None


def test_bearish_reversal_pattern_detects_shooting_star():
    df = pd.DataFrame(
        [
            {"open": 9, "high": 9.1, "low": 8, "close": 8.5},
            {"open": 10, "high": 15, "low": 9.9, "close": 9},  # shooting star
        ]
    )

    assert bearish_reversal_pattern(df, idx=1) == "shooting_star"
