"""Tests for src/data/stock_fetcher.py's pure logic (timeframe parsing,
cache path) — same convention as the rest of this project: no test here
hits the real Alpaca API (no test file exists for src/data/fetcher.py's
Binance calls either); fetch_stock_ohlcv itself is verified once, live,
against real data once API keys are available (see .env.example)."""
import pandas as pd
import pytest

from src.data.stock_fetcher import _cache_path, _drop_invalid_bars, _parse_timeframe


def test_parse_timeframe_hours():
    from alpaca.data.timeframe import TimeFrameUnit

    tf = _parse_timeframe("1h")

    assert tf.amount == 1
    assert tf.unit == TimeFrameUnit.Hour


def test_parse_timeframe_minutes():
    from alpaca.data.timeframe import TimeFrameUnit

    tf = _parse_timeframe("5m")

    assert tf.amount == 5
    assert tf.unit == TimeFrameUnit.Minute


def test_parse_timeframe_days():
    from alpaca.data.timeframe import TimeFrameUnit

    tf = _parse_timeframe("1d")

    assert tf.amount == 1
    assert tf.unit == TimeFrameUnit.Day


def test_parse_timeframe_rejects_unrecognized_string():
    with pytest.raises(ValueError):
        _parse_timeframe("bogus")


def test_cache_path_is_scoped_to_the_stock_specific_directory():
    path = _cache_path("AAPL", "1h")

    assert path.name == "AAPL_1h.parquet"
    assert path.parent.name == "ohlcv_stocks"


def test_drop_invalid_bars_removes_zero_open_and_low():
    # Real case found in GME's history (2021-06-03 13:00 UTC): a data
    # artifact from the free IEX feed, open/low printed as 0.
    df = pd.DataFrame(
        [
            {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 100.0},
            {"open": 0.0, "high": 68.58, "low": 0.0, "close": 63.63, "volume": 197096.0},
            {"open": 11.0, "high": 12.0, "low": 10.5, "close": 11.5, "volume": 200.0},
        ]
    )

    cleaned = _drop_invalid_bars(df, "GME", "1h")

    assert len(cleaned) == 2
    assert (cleaned[["open", "high", "low", "close"]] > 0).all().all()


def test_drop_invalid_bars_keeps_a_clean_dataframe_unchanged():
    df = pd.DataFrame(
        [
            {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 100.0},
            {"open": 11.0, "high": 12.0, "low": 10.5, "close": 11.5, "volume": 200.0},
        ]
    )

    cleaned = _drop_invalid_bars(df, "AAPL", "1h")

    assert len(cleaned) == 2
