"""Tests for src/backtest/walk_forward.py."""
import pandas as pd
import pytest

from src.backtest.walk_forward import chronological_folds, run_walk_forward
from src.strategies.builder import load_strategy


def test_chronological_folds_splits_into_equal_periods():
    # Boundaries aren't asserted as exact calendar-year dates: a 3-year span
    # crossing a leap year (2024) can't split into 3 identical calendar
    # years by day count. What must hold is contiguity, full coverage, and
    # roughly equal fold lengths.
    folds = chronological_folds("2023-01-01", "2026-01-01", n_folds=3)

    assert len(folds) == 3
    assert folds[0].since == "2023-01-01"
    assert folds[-1].until == "2026-01-01"
    for a, b in zip(folds, folds[1:]):
        assert a.until == b.since  # contiguous, no gaps or overlaps

    lengths_days = [
        (pd.Timestamp(f.until) - pd.Timestamp(f.since)).days for f in folds
    ]
    assert max(lengths_days) - min(lengths_days) <= 1  # equal to within a day


def test_chronological_folds_single_fold_spans_the_whole_range():
    folds = chronological_folds("2023-01-01", "2024-01-01", n_folds=1)

    assert len(folds) == 1
    assert folds[0].since == "2023-01-01"
    assert folds[0].until == "2024-01-01"


def test_chronological_folds_rejects_invalid_n_folds():
    with pytest.raises(ValueError, match="n_folds must be at least 1"):
        chronological_folds("2023-01-01", "2024-01-01", n_folds=0)


def _fake_fetch(symbol: str, timeframe: str, since, until) -> pd.DataFrame:
    # Enough flat bars to clear lookback_bars but never produce a signal —
    # this test is about fold orchestration, not backtest correctness
    # (already covered by tests/test_engine_integration.py).
    idx = pd.date_range(since, periods=250, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
        }
    )


def test_run_walk_forward_returns_one_result_per_fold():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")

    results = run_walk_forward(
        "BTC/USDT", "1h", strategy, since="2023-01-01", until="2026-01-01", n_folds=3, fetch_fn=_fake_fetch
    )

    assert len(results) == 3
    assert results[0].fold.since == "2023-01-01"
    assert results[-1].fold.until == "2026-01-01"
    for r in results:
        assert "n_trades" in r.metrics  # flat data -> 0 trades, but a well-formed metrics dict
