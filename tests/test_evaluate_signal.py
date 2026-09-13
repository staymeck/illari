"""Tests for src/backtest/engine.evaluate_signal — the single-point-in-time
check used for a live "market snapshot" (docs/PLAN.md, Phase 2), as opposed
to run_backtest's full simulation loop. Offline/synthetic only, same
convention as the rest of the suite (real-data cross-checks are done as
one-off manual scripts — see docs/PLAN.md's Verification section — not
embedded here, since that would make the suite depend on network/cache
state a fresh clone wouldn't have)."""
import pandas as pd
import pytest

from src.backtest.engine import evaluate_signal, run_backtest
from src.strategies.builder import load_strategy
from tests.test_engine_integration import _always_fires, _bar, _df, _strategy


def _flat_df(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0}
    )


def test_evaluate_signal_raises_when_not_enough_history():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    df = _flat_df(strategy.lookback_bars - 1)

    with pytest.raises(ValueError, match="needs at least"):
        evaluate_signal(df, strategy)


def test_evaluate_signal_no_signal_on_flat_data():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    df = _flat_df(strategy.lookback_bars + 5)

    result = evaluate_signal(df, strategy)

    assert result.signal is False
    assert result.reference_level is None
    assert result.as_of == df["timestamp"].iloc[-1]


def test_evaluate_signal_agrees_with_run_backtest_at_the_signal_bar():
    """Cross-check against run_backtest with a synthetic, data-dependent
    fake setup (fires only on one specific bar's close, not unconditionally
    like _always_fires) — evaluate_signal, fed only the data up to and
    including that bar (exactly what a live system would have at that
    point), must agree with what triggered run_backtest's entry."""
    from src.strategies.types import SetupResult

    def _fires_on_marker_close(ctx, params):
        if ctx.price_window["close"].iloc[-1] == 12345.0:
            return SetupResult(reference_level=100.0, extras={"setup": "marker"})
        return None

    rows = [_bar(100, 100, 100, 100)] * 3  # padding, below lookback_bars
    rows += [_bar(100, 100, 100, 100)]  # idx3: lookback satisfied, no marker yet
    rows += [_bar(100, 100.1, 12344.9, 12345.0)]  # idx4: the marker bar -> fires
    rows += [_bar(100, 100.5, 99.5, 100)] * 3  # room for the trade to resolve
    df = _df(rows)
    strategy = _strategy(_fires_on_marker_close)

    trades, _ = run_backtest(df, strategy)
    assert not trades.empty
    assert trades.iloc[0]["entry_time"] == df["timestamp"].iloc[5]  # entry = marker bar (4) + 1

    result_at_marker = evaluate_signal(df.iloc[:5], strategy)  # data through idx4 (the marker bar)
    assert result_at_marker.signal is True

    result_before_marker = evaluate_signal(df.iloc[:4], strategy)  # data through idx3 only
    assert result_before_marker.signal is False
