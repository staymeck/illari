"""Wiring tests for the Wyckoff catalog pieces — confirm each correctly
delegates to src.analysis.wyckoff (already unit-tested in
tests/test_wyckoff.py), same convention as test_strategy_pieces.py."""
import pandas as pd
import pytest

from src.strategies.risk.range_low_stop import range_low_stop
from src.strategies.risk.range_projection_target import range_projection_target
from src.strategies.setups.wyckoff_accumulation_breakout import wyckoff_accumulation_breakout
from src.strategies.types import EvalContext, SetupResult


def _candles(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows)
    df["timestamp"] = idx
    return df


def _ctx(df: pd.DataFrame) -> EvalContext:
    return EvalContext(price_window=df, marked_window=pd.DataFrame())


_RANGE_THEN_BREAKOUT_ROWS = [
    {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
    {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
    {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
    {"open": 100, "high": 102, "low": 100, "close": 101, "volume": 10},
    {"open": 100, "high": 104, "low": 100, "close": 103.6, "volume": 30},  # bullish absorption
    {"open": 101, "high": 103, "low": 99, "close": 101, "volume": 10},
    {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 10},
    {"open": 99, "high": 101, "low": 97, "close": 99, "volume": 10},
    {"open": 99, "high": 120, "low": 104, "close": 118, "volume": 50},  # breakout, above range_high=104
]


def test_wyckoff_accumulation_breakout_fires_on_a_real_breakout():
    df = _candles(_RANGE_THEN_BREAKOUT_ROWS)
    ctx = _ctx(df)

    result = wyckoff_accumulation_breakout(ctx, {"lookback": 5, "volume_window": 3})

    assert result is not None
    assert result.reference_level == pytest.approx(104.0)
    assert result.extras["range_low"] == pytest.approx(97.0)
    assert result.extras["range_high"] == pytest.approx(104.0)
    assert result.extras["accumulation_bias"] > 0


def test_wyckoff_accumulation_breakout_none_without_a_breakout():
    rows = _RANGE_THEN_BREAKOUT_ROWS[:-1] + [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}]
    df = _candles(rows)
    ctx = _ctx(df)

    assert wyckoff_accumulation_breakout(ctx, {"lookback": 5, "volume_window": 3}) is None


def test_wyckoff_accumulation_breakout_none_when_breakout_lacks_volume():
    rows = list(_RANGE_THEN_BREAKOUT_ROWS)
    rows[-1] = {"open": 99, "high": 120, "low": 104, "close": 118, "volume": 5}  # thin breakout
    df = _candles(rows)
    ctx = _ctx(df)

    assert wyckoff_accumulation_breakout(ctx, {"lookback": 5, "volume_window": 3, "breakout_volume_mult": 1.2}) is None


def test_range_low_stop_uses_the_setup_extras():
    setup = SetupResult(reference_level=104.0, extras={"range_low": 97.0})
    ctx = _ctx(_candles([{"open": 100, "close": 100, "high": 100, "low": 100, "volume": 1}]))

    stop = range_low_stop(entry_price=118.0, setup=setup, ctx=ctx, params={"buffer_pct": 1.0})

    assert stop == pytest.approx(97.0 * 0.99)


def test_range_low_stop_falls_back_without_range_low():
    setup = SetupResult(reference_level=104.0, extras={})
    ctx = _ctx(_candles([{"open": 100, "close": 100, "high": 100, "low": 100, "volume": 1}]))

    stop = range_low_stop(entry_price=118.0, setup=setup, ctx=ctx, params={"fallback_pct": 2.0})

    assert stop == pytest.approx(118.0 * 0.98)


def test_range_projection_target_matches_the_measured_move():
    df = _candles(_RANGE_THEN_BREAKOUT_ROWS)
    ctx = _ctx(df)

    # A realistic breakout entry sits just above range_high (104), with a
    # stop just below range_low (97) - not the far-away close (118) the
    # last row of _RANGE_THEN_BREAKOUT_ROWS happens to have.
    target = range_projection_target(entry_price=105.0, stop_price=96.03, ctx=ctx, params={"lookback": 5})

    # range height = 104 - 97 = 7 -> target = 105 + 7 = 112
    assert target == pytest.approx(112.0)


def test_range_projection_target_falls_back_without_a_detected_range():
    df = _candles([{"open": 100 + i * 10, "high": 105 + i * 10, "low": 100 + i * 10, "close": 104 + i * 10, "volume": 10} for i in range(6)])
    ctx = _ctx(df)

    target = range_projection_target(entry_price=150.0, stop_price=140.0, ctx=ctx, params={"lookback": 5, "fallback_ratio": 2.0})

    assert target == pytest.approx(150.0 + 2.0 * 10.0)
