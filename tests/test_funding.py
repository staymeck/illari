"""Tests for src/data/funding.py. The exchange call is monkeypatched so the
suite doesn't depend on network access; caching/pagination/date-filtering
logic is what's actually under test here."""
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.funding import fetch_funding_rate_history


def _fake_entry(ts_ms: int, rate: float) -> dict:
    return {"timestamp": ts_ms, "fundingRate": rate}


def test_fetch_funding_rate_history_filters_to_requested_range(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.funding.DATA_DIR", tmp_path)

    day_ms = 24 * 60 * 60 * 1000
    base = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    all_entries = [_fake_entry(base + i * day_ms, 0.0001 * i) for i in range(5)]

    fake_exchange = MagicMock()
    fake_exchange.fetch_funding_rate_history.side_effect = [all_entries, []]

    with patch("src.data.funding._exchange", return_value=fake_exchange):
        df = fetch_funding_rate_history(
            "BTC/USDT", since="2024-06-01", until="2024-06-03", use_cache=False
        )

    assert list(df["funding_rate"]) == [0.0, 0.0001, 0.0002]


def test_fetch_funding_rate_history_uses_cache_when_range_is_covered(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.funding.DATA_DIR", tmp_path)

    cached = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=5, freq="D", tz="UTC"),
            "funding_rate": [0.0, 0.0001, 0.0002, 0.0003, 0.0004],
        }
    )
    cache_path = tmp_path / "funding" / "BTC-USDT.parquet"
    cache_path.parent.mkdir(parents=True)
    cached.to_parquet(cache_path, index=False)

    fake_exchange = MagicMock()
    with patch("src.data.funding._exchange", return_value=fake_exchange):
        df = fetch_funding_rate_history(
            "BTC/USDT", since="2024-06-01", until="2024-06-03", use_cache=True
        )

    fake_exchange.fetch_funding_rate_history.assert_not_called()
    assert len(df) == 3
