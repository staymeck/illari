import pandas as pd
import pytest

from trading_lab.indicators.structure import find_swings, last_swing_range, trend_from_swings


def test_find_swings_detects_known_extremes_with_confirmation_delay():
    highs = [10, 15, 10, 10, 20, 10, 10]
    lows = [5, 5, 5, 1, 5, 5, 5]
    df = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "open": [(h + l) / 2 for h, l in zip(highs, lows)],
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        }
    )

    out = find_swings(df, lookback=1)

    assert list(out[out["swing_high"]].index) == [1, 4]
    assert list(out[out["swing_low"]].index) == [3]
    # The swing gets confirmed `lookback` candles after it occurs, not on its own candle.
    assert out.loc[1, "confirmed_at_index"] == 2
    assert out.loc[3, "confirmed_at_index"] == 4
    assert out.loc[4, "confirmed_at_index"] == 5


def _manual_swings_df() -> pd.DataFrame:
    # Sequence: swing low at 0 (90) -> swing high at 1 (100, confirmed at 2)
    # -> swing low at 2 (95, confirmed at 3) -> swing high at 3 (110, confirmed at 4)
    # Higher highs and higher lows -> uptrend, only visible from row 4 onward.
    return pd.DataFrame(
        {
            "high": [80, 100, 100, 110, 110, 110],
            "low": [90, 90, 95, 95, 95, 95],
            "swing_high": [False, True, False, True, False, False],
            "swing_low": [True, False, True, False, False, False],
            "confirmed_at_index": [0, 2, 3, 4, 4, 4],
        }
    )


def test_trend_from_swings_detects_uptrend_without_lookahead():
    df = _manual_swings_df()
    out = trend_from_swings(df)

    # Before having 2 confirmed swing highs and 2 confirmed swing lows, it should be 'range'.
    for i in range(4):
        assert out.loc[i, "trend"] == "range", f"row {i} should be 'range'"

    # Only at row 4 (when the 2nd swing high gets confirmed) does it become 'up'.
    assert out.loc[4, "trend"] == "up"
    assert out.loc[5, "trend"] == "up"


def test_trend_from_swings_detects_downtrend():
    df = pd.DataFrame(
        {
            "high": [110, 110, 100, 100, 90, 90],
            "low": [100, 95, 95, 85, 85, 85],
            "swing_high": [True, False, True, False, False, False],
            "swing_low": [False, True, False, True, False, False],
            "confirmed_at_index": [1, 2, 3, 4, 4, 4],
        }
    )
    out = trend_from_swings(df)
    assert out.loc[4, "trend"] == "down"


def test_last_swing_range_respects_confirmation_delay():
    df = trend_from_swings(_manual_swings_df())  # adds 'trend', doesn't affect this test

    # At as_of_index=1, the swing high on row 1 (100) is NOT confirmed yet
    # (it's confirmed on row 2) -> no swing high is visible.
    assert last_swing_range(df, as_of_index=1) is None

    # At as_of_index=2, it's already confirmed: low=90 (row 0), high=100 (row 1).
    lo, hi, direction = last_swing_range(df, as_of_index=2)
    assert (lo, hi, direction) == (pytest.approx(90), pytest.approx(100), "up")

    # At as_of_index=4, the most recently confirmed swing is low=95 (row 2) -> high=110 (row 3).
    lo, hi, direction = last_swing_range(df, as_of_index=4)
    assert (lo, hi, direction) == (pytest.approx(95), pytest.approx(110), "up")
