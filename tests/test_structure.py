"""Tests on synthetic data for src/analysis/structure.py — cases where the
expected result is known in advance (see docs/PLAN.md, Verification)."""
import pandas as pd
import pytest

from src.analysis.structure import (
    classify_trend,
    confluence_count,
    find_swing_points,
    nearest_confluent_level,
    nearest_resistance_above,
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


def test_nearest_resistance_above_picks_the_lowest_level_over_price():
    assert nearest_resistance_above(100, [80, 105, 110, 95]) == 105


def test_nearest_resistance_above_none_when_everything_is_below():
    assert nearest_resistance_above(100, [80, 90]) is None


def test_nearest_confluent_level_with_min_confluence_1_matches_plain_nearest():
    group_a = [97.0, 95.0, 80.0]
    group_b = [94.9, 70.0]
    candidates = group_a + group_b

    result = nearest_confluent_level(
        100.0, candidates, confluence_groups=(group_a, group_b), direction="below", min_confluence=1
    )

    assert result == nearest_support_below(100.0, candidates) == 97.0


def test_nearest_confluent_level_skips_a_lone_level_for_a_confirmed_one():
    # 97 is the nearest candidate but only group_a points to it (lone,
    # noise-level); 95 is a bit farther but confirmed by BOTH groups
    # (95 in group_a, 94.9 within tolerance in group_b) -> min_confluence=2
    # should skip 97 and land on 95.
    group_a = [97.0, 95.0, 80.0]
    group_b = [94.9, 70.0]
    candidates = group_a + group_b

    result = nearest_confluent_level(
        100.0, candidates, confluence_groups=(group_a, group_b), direction="below", min_confluence=2
    )

    assert result == 95.0


def test_nearest_confluent_level_none_when_nothing_qualifies():
    group_a = [95.0]
    group_b = [80.0]

    result = nearest_confluent_level(
        100.0, group_a + group_b, confluence_groups=(group_a, group_b), direction="below", min_confluence=3
    )

    assert result is None


def test_nearest_confluent_level_above_direction():
    group_a = [103.0, 105.0, 120.0]
    group_b = [105.1, 130.0]

    result = nearest_confluent_level(
        100.0, group_a + group_b, confluence_groups=(group_a, group_b), direction="above", min_confluence=2
    )

    assert result == 105.0


def test_nearest_confluent_level_rejects_bad_direction():
    with pytest.raises(ValueError):
        nearest_confluent_level(100.0, [90.0], confluence_groups=([90.0],), direction="sideways")


def test_confluence_count_counts_groups_with_a_nearby_level():
    supports = [95.0, 80.0]
    fib_levels = [95.3, 70.0]

    # 95 has a near match in both groups (95.0 and 95.3, within 1%).
    assert confluence_count(95.0, supports, fib_levels, tolerance_pct=1.0) == 2
    # 80 only matches the supports group.
    assert confluence_count(80.0, supports, fib_levels, tolerance_pct=1.0) == 1
    # An isolated level with no nearby match in either group.
    assert confluence_count(50.0, supports, fib_levels, tolerance_pct=1.0) == 0
