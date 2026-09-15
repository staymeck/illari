"""Tests for src/live/position_sizing.py's volatility-regime sizing."""
import pytest

from src.live.position_sizing import DEFAULT_MAX_MULT, DEFAULT_MIN_MULT, volatility_size_multiplier


def test_volatility_size_multiplier_full_size_at_calmest_percentile():
    assert volatility_size_multiplier(0.0) == pytest.approx(DEFAULT_MAX_MULT)


def test_volatility_size_multiplier_smallest_size_at_most_volatile_percentile():
    assert volatility_size_multiplier(100.0) == pytest.approx(DEFAULT_MIN_MULT)


def test_volatility_size_multiplier_linear_midpoint():
    # Halfway between min_mult (0.5) and max_mult (1.0) at percentile 50.
    assert volatility_size_multiplier(50.0) == pytest.approx(0.75)


def test_volatility_size_multiplier_none_treated_as_middle_of_distribution():
    assert volatility_size_multiplier(None) == pytest.approx(volatility_size_multiplier(50.0))


def test_volatility_size_multiplier_nan_treated_as_middle_of_distribution():
    assert volatility_size_multiplier(float("nan")) == pytest.approx(volatility_size_multiplier(50.0))


def test_volatility_size_multiplier_clamps_out_of_range_percentile():
    assert volatility_size_multiplier(-10.0) == pytest.approx(DEFAULT_MAX_MULT)
    assert volatility_size_multiplier(150.0) == pytest.approx(DEFAULT_MIN_MULT)


def test_volatility_size_multiplier_custom_bounds():
    assert volatility_size_multiplier(100.0, min_mult=0.25, max_mult=1.0) == pytest.approx(0.25)
    assert volatility_size_multiplier(0.0, min_mult=0.25, max_mult=1.0) == pytest.approx(1.0)
