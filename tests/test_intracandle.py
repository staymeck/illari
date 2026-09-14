"""Tests for src/analysis/intracandle.py."""
import pandas as pd
import pytest

from src.analysis.intracandle import (
    compare_groups,
    compute_intracandle_metrics,
    lag1_autocorrelation,
)


def _parents(opens_closes, freq="1h"):
    n = len(opens_closes)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": [oc[0] for oc in opens_closes],
        "close": [oc[1] for oc in opens_closes],
    })


def _children(rows, freq="5min"):
    # rows: list of (open, close) 5-min bars, evenly spaced from the same start.
    n = len(rows)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": [r[0] for r in rows],
        "close": [r[1] for r in rows],
    })


def test_efficiency_ratio_is_one_for_a_perfectly_straight_move():
    # First parent candle 100 -> 106, spanned by 3 evenly-stepping children,
    # each moving +2 with no backtracking at all. A second parent candle is
    # only there so the function can infer the candle span from the gap
    # between parent timestamps.
    parent = _parents([(100, 106), (106, 106)])
    child = _children([(100, 102), (102, 104), (104, 106)])

    result = compute_intracandle_metrics(parent, child)

    assert result.loc[0, "n_children"] == 3
    assert result.loc[0, "efficiency_ratio"] == 1.0
    assert result.loc[0, "direction_agreement_pct"] == 100.0


def test_efficiency_ratio_is_low_for_a_choppy_round_trip():
    # First parent candle barely moves (100 -> 101) but its children zigzag
    # a lot: up 10, down 9, up 0 -> most of the path is wasted motion.
    parent = _parents([(100, 101), (101, 101)])
    child = _children([(100, 110), (110, 101), (101, 101)])

    result = compute_intracandle_metrics(parent, child)

    net_change = abs(101 - 100)
    path_length = 10 + 9 + 0
    assert result.loc[0, "efficiency_ratio"] == pytest.approx(net_change / path_length)
    assert result.loc[0, "efficiency_ratio"] < 0.2


def test_parent_with_no_matching_children_gets_nan():
    parent = _parents([(100, 101), (101, 103)])
    # Children only cover the second parent's span, not the first.
    child = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 01:00", periods=2, freq="5min", tz="UTC"),
        "open": [101, 102],
        "close": [102, 103],
    })

    result = compute_intracandle_metrics(parent, child)

    assert result.loc[0, "n_children"] == 0
    assert pd.isna(result.loc[0, "efficiency_ratio"])
    assert result.loc[1, "n_children"] == 2


def test_compare_groups_detects_a_clear_difference():
    metrics = pd.DataFrame({"efficiency_ratio": [0.9, 0.85, 0.95, 0.88] + [0.1, 0.15, 0.05, 0.2]})
    is_member = pd.Series([True, True, True, True, False, False, False, False])

    report = compare_groups(metrics, is_member, "efficiency_ratio")

    assert report["n_member"] == 4
    assert report["n_rest"] == 4
    assert report["mean_member"] > report["mean_rest"]
    assert report["p_value"] < 0.05


def test_compare_groups_handles_too_few_samples():
    metrics = pd.DataFrame({"efficiency_ratio": [0.9]})
    is_member = pd.Series([True])

    report = compare_groups(metrics, is_member, "efficiency_ratio")

    assert report["n_member"] == 1
    assert report["n_rest"] == 0
    assert report["p_value"] is None


def test_lag1_autocorrelation_too_few_points_is_none():
    assert lag1_autocorrelation(pd.Series([1.0, 2.0])) is None


def test_lag1_autocorrelation_constant_series_is_none():
    # Zero variance -> correlation is undefined (NaN), not a fabricated 0.
    assert lag1_autocorrelation(pd.Series([1.0] * 10)) is None


def test_lag1_autocorrelation_sign_matches_alternating_vs_persistent():
    alternating = pd.Series([1.0, -1.0] * 10)  # every step reverses the last one
    persistent = pd.Series([1.0, 1.0, -1.0, -1.0] * 5)  # steps repeat before reversing

    assert lag1_autocorrelation(alternating) < 0
    assert lag1_autocorrelation(persistent) > 0
