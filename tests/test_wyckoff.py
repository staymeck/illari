"""Tests on synthetic data for src/analysis/wyckoff.py — expected results
known in advance (project convention, see docs/PLAN.md's Verification
section)."""
import pandas as pd
import pytest

from src.analysis.wyckoff import (
    accumulation_bias,
    close_position,
    find_trading_range,
    range_projected_target,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    out = pd.DataFrame(rows)
    out["timestamp"] = idx
    return out


def test_close_position_known_values():
    df = _df(
        [
            {"open": 100, "high": 110, "low": 100, "close": 110, "volume": 1},  # closed at the high -> 1.0
            {"open": 110, "high": 110, "low": 100, "close": 100, "volume": 1},  # closed at the low -> 0.0
            {"open": 100, "high": 110, "low": 100, "close": 105, "volume": 1},  # closed in the middle -> 0.5
        ]
    )
    pos = close_position(df)
    assert pos.iloc[0] == pytest.approx(1.0)
    assert pos.iloc[1] == pytest.approx(0.0)
    assert pos.iloc[2] == pytest.approx(0.5)


def test_find_trading_range_detects_a_tight_sideways_band():
    # 5 bars (lookback=5) tightly bounded [97, 104] -> ~7.2% range, plus a
    # 6th breakout bar that must NOT affect the detected range at all.
    rows = [
        {"open": 100, "high": 102, "low": 100, "close": 101, "volume": 10},
        {"open": 100, "high": 104, "low": 100, "close": 102, "volume": 10},
        {"open": 101, "high": 103, "low": 99, "close": 101, "volume": 10},
        {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 10},
        {"open": 99, "high": 101, "low": 97, "close": 99, "volume": 10},
        {"open": 99, "high": 115, "low": 104, "close": 114, "volume": 50},  # breakout, excluded
    ]
    df = _df(rows)

    result = find_trading_range(df, lookback=5, max_range_pct=8.0)

    assert result is not None
    assert result["high"] == pytest.approx(104.0)
    assert result["low"] == pytest.approx(97.0)
    assert result["n_bars"] == 5


def test_find_trading_range_none_when_too_volatile():
    # Strictly trending, not a range at all.
    rows = [{"open": 100 + i * 5, "high": 105 + i * 5, "low": 100 + i * 5, "close": 104 + i * 5, "volume": 10} for i in range(6)]
    df = _df(rows)

    assert find_trading_range(df, lookback=5, max_range_pct=8.0) is None


def test_find_trading_range_none_without_enough_history():
    df = _df([{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}] * 3)

    assert find_trading_range(df, lookback=5, max_range_pct=8.0) is None


def test_accumulation_bias_positive_on_bullish_absorption():
    # 3 baseline bars, then a range window with exactly one high-volume
    # bar that closes near its own high (bullish absorption / "stopping
    # volume"), the rest ordinary - net bias must come out positive.
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 102, "low": 100, "close": 101, "volume": 10},
        {"open": 100, "high": 104, "low": 100, "close": 103.6, "volume": 30},  # bullish absorption
        {"open": 101, "high": 103, "low": 99, "close": 101, "volume": 10},
        {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 10},
        {"open": 99, "high": 101, "low": 97, "close": 99, "volume": 10},
        {"open": 99, "high": 115, "low": 104, "close": 114, "volume": 50},  # breakout bar
    ]
    df = _df(rows)
    range_info = find_trading_range(df, lookback=5, max_range_pct=8.0)
    assert range_info is not None

    bias = accumulation_bias(df, range_info, volume_window=3, volume_threshold=1.5, close_position_threshold=0.3)

    assert bias == pytest.approx(0.2)  # 1 bullish, 0 bearish, out of 5 bars


def test_accumulation_bias_negative_on_bearish_absorption():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        {"open": 100, "high": 102, "low": 100, "close": 101, "volume": 10},
        {"open": 100, "high": 104, "low": 100, "close": 100.4, "volume": 30},  # bearish absorption
        {"open": 101, "high": 103, "low": 99, "close": 101, "volume": 10},
        {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 10},
        {"open": 99, "high": 101, "low": 97, "close": 99, "volume": 10},
        {"open": 99, "high": 115, "low": 104, "close": 114, "volume": 50},
    ]
    df = _df(rows)
    range_info = find_trading_range(df, lookback=5, max_range_pct=8.0)
    assert range_info is not None

    bias = accumulation_bias(df, range_info, volume_window=3, volume_threshold=1.5, close_position_threshold=0.3)

    assert bias == pytest.approx(-0.2)


def test_accumulation_bias_zero_without_any_high_volume_bars():
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}] * 9
    df = _df(rows)
    range_info = find_trading_range(df, lookback=5, max_range_pct=8.0)
    assert range_info is not None

    bias = accumulation_bias(df, range_info, volume_window=3)

    assert bias == pytest.approx(0.0)


def test_range_projected_target_long_and_short():
    assert range_projected_target(entry_price=110, range_high=104, range_low=97, direction="long") == pytest.approx(117.0)
    assert range_projected_target(entry_price=110, range_high=104, range_low=97, direction="short") == pytest.approx(103.0)


def test_range_projected_target_rejects_unknown_direction():
    with pytest.raises(ValueError):
        range_projected_target(entry_price=110, range_high=104, range_low=97, direction="sideways")
