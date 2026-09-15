"""Tests on synthetic data for src/analysis/candle_geometry.py."""
import pandas as pd

from src.analysis.candle_geometry import (
    body_ratio,
    is_inside_bar,
    is_marubozu,
    is_narrow_range,
    is_wide_range,
    lower_wick_ratio,
    upper_wick_ratio,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_body_ratio_marubozu_is_near_one():
    df = _df([{"open": 10, "high": 20, "low": 10, "close": 20}])  # no wicks at all
    assert body_ratio(df).iloc[0] == 1.0


def test_body_ratio_doji_is_near_zero():
    df = _df([{"open": 10, "high": 12, "low": 8, "close": 10}])  # body collapses to 0
    assert body_ratio(df).iloc[0] == 0.0


def test_upper_and_lower_wick_ratios():
    # range [5,15]=10; body [9,10]=1 -> upper wick 15-10=5 (50%), lower wick 9-5=4 (40%).
    df = _df([{"open": 9, "high": 15, "low": 5, "close": 10}])
    assert upper_wick_ratio(df).iloc[0] == 0.5
    assert lower_wick_ratio(df).iloc[0] == 0.4


def test_is_marubozu_true_for_full_body_candle():
    df = _df([{"open": 10, "high": 20.1, "low": 9.9, "close": 20}])
    assert is_marubozu(df).iloc[0]


def test_is_marubozu_false_for_candle_with_big_wicks():
    df = _df([{"open": 10, "high": 20, "low": 5, "close": 12}])
    assert not is_marubozu(df).iloc[0]


def test_is_narrow_range_flags_the_tightest_candle_in_the_lookback():
    # Ranges: 10,10,10,10,10,10,1 (last candle much tighter) -> NR7 at the last row.
    rows = [{"open": 0, "high": 10, "low": 0, "close": 5}] * 6 + [{"open": 5, "high": 5.5, "low": 4.5, "close": 5}]
    df = _df(rows)
    flags = is_narrow_range(df, lookback=7)
    assert not flags.iloc[5]
    assert flags.iloc[6]


def test_is_wide_range_flags_the_widest_candle_in_the_lookback():
    rows = [{"open": 0, "high": 1, "low": 0, "close": 0.5}] * 6 + [{"open": 0, "high": 20, "low": 0, "close": 10}]
    df = _df(rows)
    flags = is_wide_range(df, lookback=7)
    assert not flags.iloc[5]
    assert flags.iloc[6]


def test_is_inside_bar_true_when_fully_contained_in_prior_range():
    df = _df(
        [
            {"open": 0, "high": 20, "low": 0, "close": 10},
            {"open": 8, "high": 15, "low": 5, "close": 12},  # fully inside [0,20]
        ]
    )
    flags = is_inside_bar(df)
    assert not flags.iloc[0]  # no prior candle
    assert flags.iloc[1]


def test_is_inside_bar_false_when_it_breaks_the_prior_range():
    df = _df(
        [
            {"open": 0, "high": 20, "low": 0, "close": 10},
            {"open": 8, "high": 25, "low": 5, "close": 12},  # breaks above 20
        ]
    )
    assert not is_inside_bar(df).iloc[1]
