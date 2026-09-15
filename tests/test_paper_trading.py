"""Tests for src/live/paper_trading.py — synthetic data, same
data-dependent fake setup convention as test_lower_timeframe.py /
test_intrabar_entry.py."""
import pandas as pd
import pytest

from src.live.paper_trading import ACCOUNT_SIZES, check_market, default_state, run_check
from src.strategies.types import EvalContext, SetupResult
from tests.test_engine_integration import _bar, _df, _strategy


def _fires_above_105(ctx: EvalContext, params: dict):
    if ctx.price_window["close"].iloc[-1] >= 105:
        return SetupResult(reference_level=100.0)
    return None


def test_check_market_no_signal_stays_idle():
    df = _df([_bar(100, 100, 100, 100)] * 4)  # never crosses 105
    strategy = _strategy(_fires_above_105)

    new_state, events = check_market(df, None, strategy, None)

    assert new_state is None
    assert events == []


def test_check_market_signal_fires_goes_pending():
    df = _df([_bar(100, 100, 100, 100)] * 3 + [_bar(100, 106, 100, 106)])  # idx3 closes at 106
    strategy = _strategy(_fires_above_105)

    new_state, events = check_market(df, None, strategy, None)

    assert new_state["status"] == "pending"
    assert new_state["signal_bar_time"] == str(df["timestamp"].iloc[3])
    assert events == [{"type": "signal_pending"}]


def test_check_market_pending_waits_for_the_execution_bar_to_close():
    df = _df([_bar(100, 100, 100, 100)] * 3 + [_bar(100, 106, 100, 106)])  # only 4 rows, no bar after the signal yet
    strategy = _strategy(_fires_above_105)
    pending = {"status": "pending", "signal_bar_time": str(df["timestamp"].iloc[3])}

    new_state, events = check_market(df, None, strategy, pending)

    assert new_state == pending  # unchanged, nothing to do yet
    assert events == []


def test_check_market_pending_transitions_to_open_once_execution_bar_closes():
    df = _df(
        [_bar(100, 100, 100, 100)] * 3
        + [_bar(100, 106, 100, 106)]  # idx3: signal bar
        + [_bar(106, 106, 106, 106)]  # idx4: execution bar, opens at 106
    )
    strategy = _strategy(_fires_above_105)  # stop = 100*0.99 = 99, target = entry + 2*(entry-99)
    pending = {"status": "pending", "signal_bar_time": str(df["timestamp"].iloc[3])}

    new_state, events = check_market(df, None, strategy, pending)

    assert new_state["status"] == "open"
    assert new_state["entry_time"] == str(df["timestamp"].iloc[4])
    assert new_state["entry_price"] == 106.0
    assert new_state["stop_price"] == pytest.approx(99.0)
    assert new_state["target_price"] == pytest.approx(106.0 + 2 * (106.0 - 99.0))
    assert events[0]["type"] == "opened"


def test_check_market_open_position_does_not_prematurely_timeout():
    # max_holding_bars=2, but only 1 bar has elapsed since entry and the
    # bar is flat (no stop/target hit) - _walk_to_exit will report
    # "timeout" simply because it ran out of AVAILABLE data, not because
    # the holding window has genuinely elapsed. Must stay open.
    open_state = {
        "status": "open", "entry_time": None, "entry_price": 106.0,
        "stop_price": 99.0, "target_price": 120.0, "extras": {},
    }
    df = _df([_bar(106, 106, 106, 106)] * 2)  # entry bar + 1 more flat bar
    open_state["entry_time"] = str(df["timestamp"].iloc[0])
    strategy = _strategy(_fires_above_105, max_holding_bars=2)

    new_state, events = check_market(df, None, strategy, open_state)

    assert new_state == open_state
    assert events == []


def test_check_market_open_position_closes_on_stop_hit():
    df = _df([_bar(106, 106, 106, 106), _bar(106, 106, 90, 95)])  # 2nd bar's low breaks the stop (99)
    open_state = {
        "status": "open", "entry_time": str(df["timestamp"].iloc[0]), "entry_price": 106.0,
        "stop_price": 99.0, "target_price": 120.0, "extras": {},
    }
    strategy = _strategy(_fires_above_105, max_holding_bars=5)

    new_state, events = check_market(df, None, strategy, open_state)

    assert new_state is None
    assert events[0]["type"] == "closed"
    assert events[0]["trade"]["exit_reason"] == "stop"
    assert events[0]["trade"]["pnl_pct"] < 0


def test_check_market_open_position_closes_on_genuine_timeout():
    # max_holding_bars=1: after exactly 1 bar with no stop/target hit, this
    # IS a real timeout (bars_elapsed >= max_holding_bars).
    df = _df([_bar(106, 106, 106, 106), _bar(106, 106, 106, 106)])
    open_state = {
        "status": "open", "entry_time": str(df["timestamp"].iloc[0]), "entry_price": 106.0,
        "stop_price": 99.0, "target_price": 120.0, "extras": {},
    }
    strategy = _strategy(_fires_above_105, max_holding_bars=1)

    new_state, events = check_market(df, None, strategy, open_state)

    assert new_state is None
    assert events[0]["type"] == "closed"
    assert events[0]["trade"]["exit_reason"] == "timeout"


def test_run_check_updates_all_three_accounts_consistently():
    df = _df([_bar(106, 106, 106, 106), _bar(106, 106, 90, 95)])
    open_state = {
        "status": "open", "entry_time": str(df["timestamp"].iloc[0]), "entry_price": 106.0,
        "stop_price": 99.0, "target_price": 120.0, "extras": {},
    }
    strategy = _strategy(_fires_above_105, max_holding_bars=5)
    state = default_state()
    state["markets"]["BTC/USDT"] = open_state

    new_state, events = run_check(strategy, {"BTC/USDT": (df, None)}, state)

    closed = events[0]
    pnl_pct = closed["trade"]["pnl_pct"]
    for size in ACCOUNT_SIZES:
        key = str(int(size))
        assert closed["accounts"][key]["equity_before"] == pytest.approx(size, rel=1e-6)
        expected_after = size * (1 + pnl_pct / 100)
        # accounts[...]["equity_after"] is rounded for display; new_state's
        # own bookkeeping value is full precision.
        assert closed["accounts"][key]["equity_after"] == pytest.approx(expected_after, abs=0.01)
        assert new_state["accounts"][key] == pytest.approx(expected_after, rel=1e-6)


def test_run_check_flags_below_minimum_notional_on_open():
    df = _df([_bar(100, 100, 100, 100)] * 3 + [_bar(100, 106, 100, 106)] + [_bar(106, 106, 106, 106)])
    strategy = _strategy(_fires_above_105)
    state = default_state()
    state["accounts"]["10"] = 3.0  # already below MIN_NOTIONAL_USD (5.0)
    pending = {"status": "pending", "signal_bar_time": str(df["timestamp"].iloc[3])}
    state["markets"]["BTC/USDT"] = pending

    _, events = run_check(strategy, {"BTC/USDT": (df, None)}, state)

    opened = events[0]
    assert opened["type"] == "opened"
    assert opened["accounts"]["10"]["below_min_notional"] is True
    assert opened["accounts"]["10000"]["below_min_notional"] is False
