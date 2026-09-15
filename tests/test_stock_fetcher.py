"""Tests for src/data/stock_fetcher.py's pure logic (timeframe parsing,
cache path) — same convention as the rest of this project: no test here
hits the real Alpaca API (no test file exists for src/data/fetcher.py's
Binance calls either); fetch_stock_ohlcv itself is verified once, live,
against real data once API keys are available (see .env.example)."""
import pytest

from src.data.stock_fetcher import _cache_path, _parse_timeframe


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
