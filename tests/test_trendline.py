import math

import pandas as pd
import pytest

from trading_lab.indicators.trendline import (
    attach_trendline_features,
    distance_to_line,
    fit_line,
    line_value_at,
    trend_angle_degrees,
)


def test_fit_line_exact_points():
    slope, intercept = fit_line([0, 1, 2], [10, 12, 14])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(10.0)


def test_fit_line_requires_at_least_two_points():
    with pytest.raises(ValueError):
        fit_line([0], [10])


def test_line_value_at():
    assert line_value_at(slope=2, intercept=10, x=5) == pytest.approx(20.0)


def test_distance_to_line_above_and_below():
    # Line y = 2x + 10 -> at x=2 it's worth 14.
    assert distance_to_line(slope=2, intercept=10, x=2, y=20) == pytest.approx(6.0)
    assert distance_to_line(slope=2, intercept=10, x=2, y=10) == pytest.approx(-4.0)


@pytest.mark.parametrize("slope,expected_degrees", [(0, 0.0), (1, 45.0), (-1, -45.0)])
def test_trend_angle_degrees(slope, expected_degrees):
    assert trend_angle_degrees(slope) == pytest.approx(expected_degrees)


def _manual_structure_df() -> pd.DataFrame:
    n = 20
    df = pd.DataFrame(
        {
            "high": [1000.0] * n,
            "low": [1000.0] * n,
            "close": [1000.0] * n,
            "swing_high": [False] * n,
            "swing_low": [False] * n,
            "confirmed_at_index": [0] * n,
        }
    )
    # 4 exactly collinear swing lows: low = 1.25*idx + 97.5, each one
    # confirmed 2 candles after it occurs.
    for idx, low in [(2, 100.0), (6, 105.0), (10, 110.0), (14, 115.0)]:
        df.loc[idx, "low"] = low
        df.loc[idx, "swing_low"] = True
        df.loc[idx, "confirmed_at_index"] = idx + 2
    df.loc[16, "close"] = 136.25
    return df


def test_attach_trendline_features_no_line_before_min_points_confirmed():
    df = _manual_structure_df()
    out = attach_trendline_features(df, min_points=3, max_points=6)

    # At row 11: only the swings at idx2 (confirmed at 4) and idx6
    # (confirmed at 8) are visible -> 2 points, fewer than min_points=3.
    assert pd.isna(out.loc[11, "support_line_value"])


def test_attach_trendline_features_fits_support_line_without_lookahead():
    df = _manual_structure_df()
    out = attach_trendline_features(df, min_points=3, max_points=6)

    # By row 12 the 3rd swing low is already confirmed (idx10, confirmed at
    # 12) -> line through (2,100),(6,105),(10,110): slope 1.25, intercept 97.5.
    assert out.loc[12, "support_line_value"] == pytest.approx(1.25 * 12 + 97.5)
    assert out.loc[12, "support_angle"] == pytest.approx(math.degrees(math.atan(1.25)))

    # At row 16, with all 4 swing lows confirmed, the line stays the same
    # (they're exactly collinear) and the distance uses the manually-set close.
    assert out.loc[16, "support_line_value"] == pytest.approx(1.25 * 16 + 97.5)
    assert out.loc[16, "dist_to_support"] == pytest.approx(136.25 - (1.25 * 16 + 97.5))


def test_attach_trendline_features_resistance_is_nan_without_swing_highs():
    df = _manual_structure_df()  # no swing_high=True at all
    out = attach_trendline_features(df, min_points=3, max_points=6)
    assert out["resistance_line_value"].isna().all()
