"""Tests on synthetic data for src/analysis/structure.py — cases where the
expected result is known in advance (see docs/PLAN.md, Verification)."""
import pandas as pd

from src.analysis.structure import (
    classify_trend,
    confluence_count,
    find_swing_points,
    nearest_support_below,
    support_resistance_levels,
)


def _candles(closes: list[float]) -> pd.DataFrame:
    """Builds simple synthetic candles: high = low = open = close, with an
    incrementing hourly timestamp — enough to test pure geometry."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def test_find_swing_points_marks_known_peak_and_valley():
    # Series with a clear peak in the middle and a clear valley afterwards.
    closes = [1, 2, 3, 10, 3, 2, 1, 0.5, 1, 2, 3]
    df = _candles(closes)

    marked = find_swing_points(df, order=2)

    assert marked.loc[3, "is_swing_high"]  # the value 10
    assert marked.loc[7, "is_swing_low"]  # the value 0.5


def test_classify_trend_rising_highs_and_lows_is_uptrend():
    # Two "waves" with clearly rising highs and lows.
    closes = [1, 2, 1.5, 3, 2.5, 5, 4, 7]
    df = _candles(closes)

    assert classify_trend(df, order=1, lookback_swings=2) == "uptrend"


def test_classify_trend_falling_highs_and_lows_is_downtrend():
    closes = [7, 4, 5, 2.5, 3, 1.5, 2, 1]
    df = _candles(closes)

    assert classify_trend(df, order=1, lookback_swings=2) == "downtrend"


def test_classify_trend_without_enough_swings_is_sideways():
    closes = [1, 2, 3, 4, 5]  # monotonic trend, no intermediate pivots
    df = _candles(closes)

    assert classify_trend(df, order=2, lookback_swings=2) == "sideways"


def test_support_resistance_merges_nearby_levels():
    # Two nearly identical lows (within tolerance) should merge into just 1.
    closes = [5, 3, 5, 3.01, 5, 8, 5]
    df = _candles(closes)

    levels = support_resistance_levels(df, order=1, tolerance_pct=1.0)

    assert len(levels["support"]) == 1
    assert 3.0 <= levels["support"][0] <= 3.01


def test_nearest_support_below_picks_the_highest_level_under_price():
    assert nearest_support_below(100, [80, 90, 95, 105]) == 95


def test_nearest_support_below_none_when_everything_is_above():
    assert nearest_support_below(100, [105, 110]) is None


def test_nearest_support_below_empty_levels():
    assert nearest_support_below(100, []) is None


def test_confluence_count_counts_groups_with_a_nearby_level():
    supports = [95.0, 80.0]
    fib_levels = [95.3, 70.0]

    # 95 has a near match in both groups (95.0 and 95.3, within 1%).
    assert confluence_count(95.0, supports, fib_levels, tolerance_pct=1.0) == 2
    # 80 only matches the supports group.
    assert confluence_count(80.0, supports, fib_levels, tolerance_pct=1.0) == 1
    # An isolated level with no nearby match in either group.
    assert confluence_count(50.0, supports, fib_levels, tolerance_pct=1.0) == 0
