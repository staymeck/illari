"""Tests for src/data/fear_greed.py. The HTTP call is monkeypatched so the
suite doesn't depend on network access."""
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.fear_greed import fetch_fear_greed_index


def _fake_response(entries: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"data": entries}
    response.raise_for_status.return_value = None
    return response


def test_fetch_fear_greed_index_parses_and_filters(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.fear_greed.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.data.fear_greed._CACHE_FILE", tmp_path / "fear_greed" / "fear_greed_index.parquet")

    day_s = 24 * 60 * 60
    base = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp())
    entries = [
        {"value": str(50 + i), "value_classification": "Neutral", "timestamp": str(base + i * day_s)}
        for i in range(5)
    ]

    with patch("src.data.fear_greed.requests.get", return_value=_fake_response(entries)) as mock_get:
        df = fetch_fear_greed_index(since="2024-06-01", until="2024-06-03", use_cache=False)

    mock_get.assert_called_once()
    assert list(df["value"]) == [50, 51, 52]
    assert set(df["classification"]) == {"Neutral"}


def test_fetch_fear_greed_index_uses_cache_when_range_is_covered(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.fear_greed.DATA_DIR", tmp_path)
    cache_file = tmp_path / "fear_greed" / "fear_greed_index.parquet"
    monkeypatch.setattr("src.data.fear_greed._CACHE_FILE", cache_file)

    cached = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=5, freq="D", tz="UTC"),
            "value": [50, 51, 52, 53, 54],
            "classification": ["Neutral"] * 5,
        }
    )
    cache_file.parent.mkdir(parents=True)
    cached.to_parquet(cache_file, index=False)

    with patch("src.data.fear_greed.requests.get") as mock_get:
        df = fetch_fear_greed_index(since="2024-06-01", until="2024-06-03", use_cache=True)

    mock_get.assert_not_called()
    assert len(df) == 3
