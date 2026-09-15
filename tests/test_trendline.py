"""Tests on synthetic data for src/analysis/trendline.py."""
import pandas as pd
import pytest

from src.analysis.structure import find_swing_points
from src.analysis.trendline import (
    distance_to_line,
    fit_line,
    fit_swing_trendline,
    line_value_at,
    trend_angle_degrees,
    trendline_distance,
)


def _candles(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx, "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1.0] * len(closes),
        }
    )


def test_fit_line_exact_two_points():
    slope, intercept = fit_line([0.0, 2.0], [10.0, 14.0])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(10.0)


def test_fit_line_requires_at_least_two_points():
    with pytest.raises(ValueError):
        fit_line([1.0], [5.0])


def test_line_value_at_and_distance_to_line():
    slope, intercept = 2.0, 10.0
    assert line_value_at(slope, intercept, 5.0) == pytest.approx(20.0)
    # Price 25 sits 5 above the line's value (20) at x=5.
    assert distance_to_line(slope, intercept, 5.0, 25.0) == pytest.approx(5.0)


def test_trend_angle_degrees_flat_and_45_degrees():
    assert trend_angle_degrees(0.0) == pytest.approx(0.0)
    assert trend_angle_degrees(1.0) == pytest.approx(45.0)


def test_fit_swing_trendline_rising_swing_lows():
    # Rising swing lows at index 1 (low=5), 3 (low=6), 5 (low=7) with
    # order=1: peaks in between confirm each as a real swing low.
    closes = [10, 5, 10, 6, 10, 7, 10]
    df = _candles(closes)
    marked = find_swing_points(df, order=1)

    fit = fit_swing_trendline(marked, kind="low", min_points=3)

    assert fit is not None
    slope, _intercept = fit
    assert slope > 0  # rising support line


def test_fit_swing_trendline_none_when_not_enough_swings():
    closes = [10, 5, 10]
    df = _candles(closes)
    marked = find_swing_points(df, order=1)

    assert fit_swing_trendline(marked, kind="low", min_points=3) is None


def test_fit_swing_trendline_missing_column_returns_none():
    assert fit_swing_trendline(pd.DataFrame({"low": [1, 2, 3]}), kind="low") is None


def test_trendline_distance_positive_above_negative_below():
    closes = [10, 5, 10, 6, 10, 7, 10]
    df = _candles(closes)
    marked = find_swing_points(df, order=1)

    above = trendline_distance(marked, current_index=6.0, current_price=100.0, kind="low", min_points=3)
    below = trendline_distance(marked, current_index=6.0, current_price=1.0, kind="low", min_points=3)

    assert above is not None and above > 0
    assert below is not None and below < 0


def test_trendline_distance_none_without_a_fitted_line():
    df = _candles([10, 5, 10])
    marked = find_swing_points(df, order=1)
    assert trendline_distance(marked, current_index=2.0, current_price=50.0, kind="low", min_points=3) is None
