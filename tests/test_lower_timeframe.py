"""Tests for the lower-timeframe (finer-grained) confirmation feature:
engine.py's run_backtest(lower_tf_df=...) wiring and
confirmations/lower_tf_confirmation.py."""
from dataclasses import replace

import pandas as pd

from src.backtest.engine import run_backtest
from src.strategies.builder import ResolvedConfirmation
from src.strategies.confirmations.lower_tf_confirmation import lower_tf_rejection, lower_tf_structure_break
from src.strategies.types import ConfirmationResult, EvalContext
from tests.test_engine_integration import _always_fires, _bar, _df, _strategy


def _row(open_, high, low, close) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def _lower_tf_window(rows: list[tuple]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(
        [{"timestamp": t, "open": o, "high": h, "low": l, "close": c} for t, (o, h, l, c) in zip(idx, rows)]
    )


# --- lower_tf_rejection ---


def test_lower_tf_rejection_fires_on_a_hammer_shaped_last_candle():
    window = _lower_tf_window([(10, 10.2, 9.8, 10), (9, 10.1, 5, 10)])  # last candle: hammer shape

    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=window)
    result = lower_tf_rejection(ctx, {})

    assert result is not None
    assert result.extras["lower_tf_rejection"] is True


def test_lower_tf_rejection_none_for_a_plain_last_candle():
    window = _lower_tf_window([(10, 10.2, 9.8, 10), (10, 10.5, 9.5, 10.2)])

    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=window)

    assert lower_tf_rejection(ctx, {}) is None


def test_lower_tf_rejection_none_without_lower_tf_data():
    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=None)

    assert lower_tf_rejection(ctx, {}) is None


# --- lower_tf_structure_break ---


def test_lower_tf_structure_break_fires_above_prior_high():
    highs = [10, 11, 9, 10, 10]
    rows = [(h, h, h, h) for h in highs] + [(14, 15, 14, 15)]  # last candle breaks above 11
    window = _lower_tf_window(rows)

    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=window)
    result = lower_tf_structure_break(ctx, {"lookback": 5})

    assert result is not None
    assert result.extras["lower_tf_break_level"] == 11


def test_lower_tf_structure_break_none_when_not_breaking():
    highs = [10, 11, 9, 10, 10]
    rows = [(h, h, h, h) for h in highs] + [(10.5, 10.9, 10.4, 10.5)]  # doesn't clear 11
    window = _lower_tf_window(rows)

    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=window)

    assert lower_tf_structure_break(ctx, {"lookback": 5}) is None


def test_lower_tf_structure_break_none_without_enough_history():
    window = _lower_tf_window([(10, 10, 10, 10), (11, 11, 11, 11)])

    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=window)

    assert lower_tf_structure_break(ctx, {"lookback": 5}) is None


def test_lower_tf_structure_break_none_without_lower_tf_data():
    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), lower_tf_window=None)

    assert lower_tf_structure_break(ctx, {}) is None


# --- engine wiring: run_backtest(lower_tf_df=...) ---


def test_run_backtest_lower_tf_window_uses_the_bars_close_not_its_open():
    captured: list = []

    def _probe(ctx: EvalContext, params: dict) -> ConfirmationResult:
        closes = None if ctx.lower_tf_window is None else list(ctx.lower_tf_window["close"])
        captured.append(closes)
        return ConfirmationResult(name="probe")

    parent = _df([_bar(100, 100, 100, 100)] * 5)  # lookback_bars=3 -> only bar idx3 gets evaluated
    strategy = _strategy(_always_fires)
    strategy = replace(strategy, confirmations=[ResolvedConfirmation(name="probe", fn=_probe, params={})])

    # Bar idx3 opens 2024-01-01 03:00 and closes 04:00. 5-min candles from
    # 02:00 to 04:00 inclusive (25 rows, close = row index): the 03:55 one
    # (value 23) closes exactly at 04:00 and must be included; the 04:00
    # one (value 24) closes at 04:05, after the bar's own close, and must
    # NOT be included yet.
    ts = pd.date_range("2024-01-01 02:00", "2024-01-01 04:00", freq="5min", tz="UTC")
    lower_tf = pd.DataFrame({"timestamp": ts, "open": range(len(ts)), "high": range(len(ts)),
                              "low": range(len(ts)), "close": list(range(len(ts)))})

    run_backtest(parent, strategy, lower_tf_df=lower_tf, lower_tf_lookback_bars=10)

    assert captured[0][-1] == 23
    assert 24 not in captured[0]
    assert len(captured[0]) == 10


def test_run_backtest_without_lower_tf_df_is_unaffected():
    from src.strategies.builder import load_strategy

    idx = pd.date_range("2024-01-01", periods=250, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": idx, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0}
    )
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")

    trades_a, equity_a = run_backtest(df, strategy)
    trades_b, equity_b = run_backtest(df, strategy, lower_tf_df=None)

    assert trades_a.equals(trades_b)
    assert equity_a.equals(equity_b)
