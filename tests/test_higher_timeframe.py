"""Tests for the multi-timeframe confirmation feature: engine.py's
_higher_tf_window_as_of / run_backtest(higher_tf_df=...), and
confirmations/higher_tf_trend.py."""
import pandas as pd

from src.backtest.engine import _higher_tf_window_as_of, run_backtest
from src.strategies.confirmations.higher_tf_trend import higher_tf_trend
from src.strategies.types import EvalContext


def _daily_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "open": closes, "high": closes, "low": closes, "close": closes, "volume": [1.0] * len(closes)}
    )


def test_higher_tf_window_as_of_excludes_the_still_forming_candle():
    # Daily candles at 00:00 each day; interval = 1 day.
    daily = _daily_df([1, 2, 3, 4, 5])
    interval = pd.Timedelta(days=1)

    # A lower-timeframe bar at 2024-01-03 05:00: the Jan-3 daily candle
    # opened at Jan-3 00:00 but doesn't close until Jan-4 00:00, so it must
    # NOT be included yet.
    window = _higher_tf_window_as_of(daily, pd.Timestamp("2024-01-03 05:00", tz="UTC"), lookback_bars=10, interval=interval)

    assert list(window["close"]) == [1, 2]  # only Jan-1 and Jan-2 have fully closed


def test_higher_tf_window_as_of_includes_a_just_closed_candle():
    daily = _daily_df([1, 2, 3, 4, 5])
    interval = pd.Timedelta(days=1)

    # Exactly at the Jan-4 00:00 open: the Jan-3 candle (opened Jan-3 00:00)
    # has now closed.
    window = _higher_tf_window_as_of(daily, pd.Timestamp("2024-01-04 00:00", tz="UTC"), lookback_bars=10, interval=interval)

    assert list(window["close"]) == [1, 2, 3]


def test_higher_tf_window_as_of_respects_lookback_bars():
    daily = _daily_df(list(range(1, 21)))
    interval = pd.Timedelta(days=1)

    window = _higher_tf_window_as_of(daily, pd.Timestamp("2024-01-20 00:00", tz="UTC"), lookback_bars=5, interval=interval)

    assert len(window) == 5
    assert list(window["close"]) == [15, 16, 17, 18, 19]


def test_higher_tf_window_as_of_empty_before_any_candle_closed():
    daily = _daily_df([1, 2, 3])
    interval = pd.Timedelta(days=1)

    window = _higher_tf_window_as_of(daily, pd.Timestamp("2024-01-01 05:00", tz="UTC"), lookback_bars=10, interval=interval)

    assert window.empty


def test_higher_tf_trend_confirmation_none_without_higher_tf_data():
    ctx = EvalContext(price_window=pd.DataFrame(), marked_window=pd.DataFrame(), higher_tf_window=None)

    assert higher_tf_trend(ctx, {}) is None


def test_higher_tf_trend_confirmation_matches_required_trend():
    uptrend_closes = [1, 2, 1.5, 3, 2.5, 5, 4, 7]  # same series used in test_structure.py's uptrend test
    ctx = EvalContext(
        price_window=pd.DataFrame(),
        marked_window=pd.DataFrame(),
        higher_tf_window=_daily_df(uptrend_closes),
    )

    result = higher_tf_trend(ctx, {"required": "uptrend", "order": 1, "lookback_swings": 2})

    assert result is not None
    assert result.extras["higher_tf_trend"] == "uptrend"


def test_higher_tf_trend_confirmation_none_when_mismatched():
    downtrend_closes = [7, 4, 5, 2.5, 3, 1.5, 2, 1]
    ctx = EvalContext(
        price_window=pd.DataFrame(),
        marked_window=pd.DataFrame(),
        higher_tf_window=_daily_df(downtrend_closes),
    )

    result = higher_tf_trend(ctx, {"required": "uptrend", "order": 1, "lookback_swings": 2})

    assert result is None


def test_run_backtest_without_higher_tf_df_is_unaffected():
    """Regression guard: omitting higher_tf_df must be byte-for-byte
    identical to a plain run_backtest call — see the real regression check
    against BTC/USDT run separately (not in this offline test)."""
    from src.strategies.builder import load_strategy

    idx = pd.date_range("2024-01-01", periods=250, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": idx, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0}
    )
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")

    trades_a, equity_a = run_backtest(df, strategy)
    trades_b, equity_b = run_backtest(df, strategy, higher_tf_df=None)

    assert trades_a.equals(trades_b)
    assert equity_a.equals(equity_b)
