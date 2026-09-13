"""Integration tests for src/backtest/engine.run_backtest — the actual
simulation loop (entry timing, stop/target/timeout exits, no overlapping
trades, fee/equity math) had NO automated coverage before this: confidence
came only from manually diffing against real-market numbers recorded during
this session, not from the pytest suite. These tests close that gap using
fully deterministic, hand-built pieces (not real market data) so the
expected outcome is known exactly in advance.
"""
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.strategies.builder import ResolvedStrategy
from src.strategies.risk.fixed_pct import fixed_pct_stop, risk_reward_target
from src.strategies.types import EvalContext, SetupResult

_LOOKBACK_BARS = 3
_REFERENCE_LEVEL = 100.0


def _always_uptrend(ctx: EvalContext, params: dict) -> str:
    return "uptrend"


def _always_fires(ctx: EvalContext, params: dict) -> SetupResult | None:
    return SetupResult(reference_level=_REFERENCE_LEVEL, extras={"setup": "fake"})


def _never_fires(ctx: EvalContext, params: dict) -> SetupResult | None:
    return None


def _strategy(setup_fn, max_holding_bars: int = 10) -> ResolvedStrategy:
    """A minimal strategy with no confirmations, built from fake context/setup
    pieces plus the real (already unit-tested) fixed_pct/risk_reward risk
    pieces — isolates the engine's loop mechanics from piece-resolution
    logic, which is covered separately in test_builder.py / test_strategy_pieces.py."""
    return ResolvedStrategy(
        name="test_strategy",
        lookback_bars=_LOOKBACK_BARS,
        swing_order=1,
        context_fn=_always_uptrend,
        context_params={},
        required_context="uptrend",
        setup_fn=setup_fn,
        setup_params={},
        confirmations=[],
        stop_fn=fixed_pct_stop,
        stop_params={"pct_below_reference": 1.0},  # stop = 100 * 0.99 = 99
        target_fn=risk_reward_target,
        target_params={"ratio": 2.0},  # target = entry + 2 * risk
        max_holding_bars=max_holding_bars,
        fee_pct=0.1,
        initial_equity=10_000.0,
    )


def _bar(open_, high, low, close) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC"))
    return df


def test_run_backtest_exits_on_target_hit():
    # 3 padding bars (lookback) + entry bar (no exit) + target-hit bar +
    # 1 trailing bar so the loop ends exactly after the one trade.
    df = _df(
        [
            _bar(100, 100, 100, 100),  # idx0 padding
            _bar(100, 100, 100, 100),  # idx1
            _bar(100, 100, 100, 100),  # idx2
            _bar(100, 100, 100, 100),  # idx3: setup fires here, entry next bar
            _bar(100, 100.5, 99.5, 100),  # idx4: entry bar, no stop/target hit
            _bar(100, 102.5, 99.6, 101),  # idx5: high >= target(102) -> exit
            _bar(101, 101, 101, 101),  # idx6: trailing, loop ends (i < n-1 fails)
        ]
    )

    trades, equity_curve = run_backtest(df, _strategy(_always_fires))

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 102.0  # the target price, not the bar's actual high
    assert trade["exit_reason"] == "target"
    assert trade["setup"] == "fake"  # setup extras merged onto the trade
    gross_pnl_pct = (102.0 - 100.0) / 100.0 * 100
    expected_net_pct = gross_pnl_pct - 2 * 0.1
    assert trade["pnl_pct"] == expected_net_pct
    assert equity_curve.iloc[-1] == pytest.approx(10_000.0 * (1 + expected_net_pct / 100))


def test_run_backtest_exits_on_stop_hit():
    df = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100.5, 99.5, 100),  # idx4: entry, no hit
            _bar(100, 100.5, 98.5, 99),  # idx5: low <= stop(99) -> exit
            _bar(99, 99, 99, 99),
        ]
    )

    trades, _ = run_backtest(df, _strategy(_always_fires))

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 99.0  # the stop price, not the bar's actual low
    assert trade["pnl_pct"] < 0


def test_run_backtest_exits_on_timeout_when_neither_stop_nor_target_hit():
    df = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100.5, 99.5, 100),  # idx4: entry
            _bar(100, 100.5, 99.5, 100.2),  # idx5: neither hit -> timeout (max_holding_bars=1)
            _bar(100, 100, 100, 100),
        ]
    )

    trades, _ = run_backtest(df, _strategy(_always_fires, max_holding_bars=1))

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "timeout"
    assert trade["exit_price"] == 100.2  # closing price of the last held bar


def test_run_backtest_no_signal_produces_no_trades():
    df = _df([_bar(100, 100, 100, 100)] * 10)

    trades, equity_curve = run_backtest(df, _strategy(_never_fires))

    assert trades.empty
    assert (equity_curve == 10_000.0).all()


def test_run_backtest_trades_do_not_overlap():
    # Setup always fires; each completed trade must start strictly after the
    # previous one's exit bar (never at or before it).
    df = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 102.5, 99.5, 100),  # idx4: entry, target hit immediately
            _bar(100, 100, 100, 100),
            _bar(100, 102.5, 99.5, 100),  # idx6: next entry, target hit immediately
            _bar(100, 100, 100, 100),
        ]
    )

    trades, _ = run_backtest(df, _strategy(_always_fires))

    entry_times = list(trades["entry_time"])
    exit_times = list(trades["exit_time"])
    for k in range(1, len(entry_times)):
        assert entry_times[k] > exit_times[k - 1]
