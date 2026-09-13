"""Tests on synthetic data for src/analysis/fibonacci.py."""
import pandas as pd

from src.analysis.fibonacci import (
    is_near_confluence,
    latest_up_leg,
    nearest_level,
    retracement_levels,
)


def _candles(closes: list[float]) -> pd.DataFrame:
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


def test_retracement_levels_known_values():
    levels = retracement_levels(swing_low=100.0, swing_high=200.0)

    assert levels[0.5] == 150.0
    assert levels[0.618] == 200.0 - 100.0 * 0.618
    assert levels[0.236] == 200.0 - 100.0 * 0.236


def test_retracement_levels_rejects_inverted_leg():
    try:
        retracement_levels(swing_low=200.0, swing_high=100.0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_nearest_level_picks_closest_ratio():
    levels = retracement_levels(swing_low=0.0, swing_high=100.0)  # 0.618 -> 38.2

    ratio, level_price = nearest_level(40.0, levels)

    assert ratio == 0.618
    assert level_price == 38.2


def test_is_near_confluence_within_tolerance():
    # leg from 0 to 100: the 61.8% retracement sits at price 38.2
    match = is_near_confluence(price=38.3, swing_low=0.0, swing_high=100.0, tolerance_pct=1.0)

    assert match is not None
    ratio, level_price = match
    assert ratio == 0.618


def test_is_near_confluence_outside_tolerance_returns_none():
    match = is_near_confluence(price=70.0, swing_low=0.0, swing_high=100.0, tolerance_pct=0.5)

    assert match is None


def test_latest_up_leg_finds_low_then_high():
    # Clear low at index 3, clear high at index 7 (later in time).
    closes = [5, 4, 3, 1, 3, 5, 7, 10, 8, 6]
    df = _candles(closes)

    leg = latest_up_leg(df, order=2)

    assert leg is not None
    assert leg["low"] == 1
    assert leg["high"] == 10
    assert leg["low_idx"] < leg["high_idx"]


def test_latest_up_leg_returns_none_when_last_pivot_is_a_low():
    # Price makes a high then pulls back to a new low with no higher high yet.
    closes = [1, 5, 10, 7, 5, 3, 1, 0.5, 1, 2]
    df = _candles(closes)

    leg = latest_up_leg(df, order=2)

    assert leg is None


def test_latest_up_leg_returns_none_when_low_precedes_high_but_is_not_lower():
    # Regression test: found during the 5-market x 36-month scale-up run.
    # A swing low can chronologically precede a swing high while still
    # sitting ABOVE that high's price — e.g. within a trimmed lookback
    # window during an overall downtrend, the most recent *confirmed* low
    # can be numerically higher than the most recent *confirmed* high. That
    # is not a genuine up-leg, and must not be treated as one (it used to
    # reach retracement_levels() with swing_high < swing_low and raise
    # ValueError). Built directly as a `marked` frame to target this exact
    # code path regardless of how the underlying pivots were produced.
    marked = pd.DataFrame(
        {
            "high": [100, 80, 70, 60],
            "low": [100, 80, 70, 60],
            "is_swing_low": [False, True, False, False],  # idx1, price 80
            "is_swing_high": [False, False, False, True],  # idx3, price 60 — lower, but later
        }
    )

    leg = latest_up_leg(pd.DataFrame(), order=2, marked=marked)

    assert leg is None
