"""Tests on synthetic data for src/backtest/engine.py."""
import pandas as pd

from src.analysis.structure import find_swing_points
from src.backtest.engine import _confirmed_pivots_as_of, compute_metrics


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


def test_confirmed_pivots_as_of_hides_pivot_before_confirmation_bar():
    # order=2 -> a pivot at index 5 needs 2 bars after it (index 7) to confirm.
    closes = [5, 4, 3, 1, 3, 0.5, 1, 2, 3, 4]  # swing low at index 5 (value 0.5)
    marked_full = find_swing_points(_candles(closes), order=2)

    # One bar too early: the pivot must not be visible yet.
    too_early = _confirmed_pivots_as_of(marked_full, i=6, window_start=0, swing_order=2)
    assert not too_early["is_swing_low"].any()

    # Exactly at the confirmation bar (5 + 2 = 7): now it must be visible.
    on_time = _confirmed_pivots_as_of(marked_full, i=7, window_start=0, swing_order=2)
    assert on_time["is_swing_low"].any()


def test_confirmed_pivots_as_of_respects_window_start():
    closes = [5, 4, 3, 1, 3, 0.5, 1, 2, 3, 4]
    marked_full = find_swing_points(_candles(closes), order=2)

    # Even though bar 7 would confirm it, a window starting after the pivot
    # (e.g. a short lookback) must not see it either.
    sliced_out = _confirmed_pivots_as_of(marked_full, i=7, window_start=6, swing_order=2)
    assert sliced_out.empty


def test_compute_metrics_known_trades():
    # 2 winners (+100 each) and 1 loser (-50): result known by hand.
    trades = pd.DataFrame(
        {
            "pnl_abs": [100.0, 100.0, -50.0],
            "equity_after": [10100.0, 10200.0, 10150.0],
        }
    )
    equity_curve = pd.Series([10000.0, 10100.0, 10200.0, 10150.0])

    metrics = compute_metrics(trades, equity_curve, initial_equity=10000.0)

    assert metrics["n_trades"] == 3
    assert metrics["win_rate_pct"] == round(2 / 3 * 100, 2)
    assert metrics["profit_factor"] == round(200 / 50, 2)
    assert metrics["expectancy"] == round((100 + 100 - 50) / 3, 2)
    assert metrics["final_equity"] == 10150.0
    assert metrics["total_return_pct"] == 1.5
    # drawdown from the 10200 peak down to 10150 = -0.49%
    assert metrics["max_drawdown_pct"] == round((10150 - 10200) / 10200 * 100, 2)


def test_compute_metrics_no_trades_returns_zeros():
    metrics = compute_metrics(pd.DataFrame(), pd.Series([10000.0]), initial_equity=10000.0)

    assert metrics["n_trades"] == 0
    assert metrics["final_equity"] == 10000.0
