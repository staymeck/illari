"""Tests on synthetic data for src/analysis/candles.py."""
import pandas as pd

from src.analysis.candles import (
    bearish_reversal_pattern,
    bullish_reversal_pattern,
    doji_pattern,
    is_bearish_engulfing,
    is_bearish_harami,
    is_bullish_engulfing,
    is_bullish_harami,
    is_dark_cloud_cover,
    is_doji,
    is_dragonfly_doji,
    is_evening_star,
    is_gravestone_doji,
    is_hammer,
    is_long_legged_doji,
    is_morning_star,
    is_piercing_pattern,
    is_shooting_star,
    is_three_black_crows,
    is_three_white_soldiers,
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


def test_bearish_reversal_pattern_detects_hanging_man_shape():
    # Identical shape to a hammer - bearish_reversal_pattern must label it
    # "hanging_man" instead, since only trend context tells them apart.
    df = pd.DataFrame([{"open": 9, "high": 10.1, "low": 5, "close": 10}])

    assert bearish_reversal_pattern(df, idx=0) == "hanging_man"


def test_bullish_reversal_pattern_detects_inverted_hammer_shape():
    # Identical shape to a shooting star.
    df = pd.DataFrame([{"open": 10, "high": 15, "low": 9.9, "close": 9}])

    assert bullish_reversal_pattern(df, idx=0) == "inverted_hammer"


# --- Doji variants ---


def test_is_gravestone_doji_true_for_tiny_body_and_long_upper_wick():
    row = _row(open_=10, high=11, low=9.95, close=10)

    assert is_gravestone_doji(row)
    assert not is_dragonfly_doji(row)
    assert doji_pattern(row) == "gravestone_doji"


def test_is_dragonfly_doji_true_for_tiny_body_and_long_lower_wick():
    row = _row(open_=10, high=10.05, low=9, close=10)

    assert is_dragonfly_doji(row)
    assert not is_gravestone_doji(row)
    assert doji_pattern(row) == "dragonfly_doji"


def test_is_long_legged_doji_true_for_tiny_body_and_both_wicks_large():
    row = _row(open_=10, high=11, low=9, close=10)

    assert is_long_legged_doji(row)
    assert doji_pattern(row) == "long_legged_doji"


def test_doji_pattern_generic_for_asymmetric_but_not_extreme_wicks():
    row = _row(open_=10, high=10.8, low=9.8, close=10)  # upper ratio .8, lower ratio .2

    assert doji_pattern(row) == "doji"


def test_doji_pattern_none_for_a_large_body_candle():
    row = _row(open_=10, high=10.5, low=9.5, close=10.9)

    assert doji_pattern(row) is None


# --- Two-candle patterns ---


def test_is_bullish_harami_true_for_a_small_body_contained_inside():
    prev_row = _row(open_=12, high=12.2, low=8.8, close=9)  # large bearish
    row = _row(open_=10, high=11.2, low=9.8, close=11)  # small bullish, contained

    assert is_bullish_harami(prev_row, row)


def test_is_bearish_harami_true_for_a_small_body_contained_inside():
    prev_row = _row(open_=9, high=12.2, low=8.8, close=12)  # large bullish
    row = _row(open_=11, high=11.2, low=9.8, close=10)  # small bearish, contained

    assert is_bearish_harami(prev_row, row)


def test_is_piercing_pattern_true():
    prev_row = _row(open_=12, high=12.1, low=8.9, close=9)  # large bearish, midpoint 10.5
    row = _row(open_=8.5, high=11.1, low=8.4, close=11)  # gaps down, closes past midpoint

    assert is_piercing_pattern(prev_row, row)


def test_is_dark_cloud_cover_true():
    prev_row = _row(open_=9, high=12.1, low=8.9, close=12)  # large bullish, midpoint 10.5
    row = _row(open_=12.5, high=12.6, low=9.9, close=10)  # gaps up, closes past midpoint

    assert is_dark_cloud_cover(prev_row, row)


# --- Three-candle patterns ---


def test_is_morning_star_true():
    df = pd.DataFrame([
        {"open": 12, "high": 12.1, "low": 8.9, "close": 9},   # large bearish, midpoint 10.5
        {"open": 9, "high": 9.1, "low": 8.7, "close": 8.8},   # small indecision body
        {"open": 9, "high": 11.1, "low": 8.9, "close": 11},   # large bullish, closes past midpoint
    ])

    assert is_morning_star(df, idx=2)
    assert bullish_reversal_pattern(df, idx=2) == "morning_star"


def test_is_evening_star_true():
    df = pd.DataFrame([
        {"open": 9, "high": 12.1, "low": 8.9, "close": 12},    # large bullish, midpoint 10.5
        {"open": 12, "high": 12.3, "low": 11.9, "close": 12.2},  # small indecision body
        {"open": 12, "high": 12.1, "low": 9.9, "close": 10},   # large bearish, closes past midpoint
    ])

    assert is_evening_star(df, idx=2)
    assert bearish_reversal_pattern(df, idx=2) == "evening_star"


def test_is_three_white_soldiers_true():
    df = pd.DataFrame([
        {"open": 9, "high": 10.1, "low": 8.9, "close": 10},
        {"open": 10, "high": 11.1, "low": 9.9, "close": 11},
        {"open": 11, "high": 12.1, "low": 10.9, "close": 12},
    ])

    assert is_three_white_soldiers(df, idx=2)


def test_is_three_black_crows_true():
    df = pd.DataFrame([
        {"open": 12, "high": 12.1, "low": 10.9, "close": 11},
        {"open": 11, "high": 11.1, "low": 9.9, "close": 10},
        {"open": 10, "high": 10.1, "low": 8.9, "close": 9},
    ])

    assert is_three_black_crows(df, idx=2)


def test_three_candle_patterns_false_with_not_enough_history():
    df = pd.DataFrame([{"open": 9, "high": 10.1, "low": 8.9, "close": 10}])

    assert is_morning_star(df, idx=0) is False
    assert is_three_white_soldiers(df, idx=0) is False
