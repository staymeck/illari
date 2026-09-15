"""Tests for src/analysis/llm_score.py — offline only, per the project's
convention (a live Ollama server is a manual-script concern, not something
the suite depends on). The HTTP call is mocked."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from src.analysis.llm_score import format_candles, parse_score, score_candles


def _window() -> pd.DataFrame:
    return pd.DataFrame({"open": [100.0, 100.5], "high": [100.6, 101.0], "low": [99.8, 100.2], "close": [100.5, 100.9]})


def test_format_candles_lists_ohlc_oldest_to_newest():
    text = format_candles(_window())

    lines = text.splitlines()
    assert len(lines) == 2
    assert "O=100.0000" in lines[0] and "C=100.5000" in lines[0]
    assert "O=100.5000" in lines[1] and "C=100.9000" in lines[1]


def test_parse_score_extracts_a_valid_integer():
    assert parse_score("73") == 73
    assert parse_score(" 42 \n") == 42
    assert parse_score("Score: 15") == 15


def test_parse_score_none_for_out_of_range_or_unparseable():
    assert parse_score("150") is None  # out of 0-100 range
    assert parse_score("not a number") is None
    assert parse_score("") is None


def test_score_candles_returns_parsed_score_on_success():
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"response": "68"}).encode()
    fake_response.__enter__.return_value = fake_response

    with patch("src.analysis.llm_score.urllib.request.urlopen", return_value=fake_response):
        score = score_candles(_window())

    assert score == 68


def test_score_candles_none_when_the_server_is_unreachable():
    with patch("src.analysis.llm_score.urllib.request.urlopen", side_effect=OSError("connection refused")):
        score = score_candles(_window())

    assert score is None


def test_score_candles_none_on_unparseable_response():
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"response": "I think it looks bullish overall"}).encode()
    fake_response.__enter__.return_value = fake_response

    with patch("src.analysis.llm_score.urllib.request.urlopen", return_value=fake_response):
        score = score_candles(_window())

    assert score is None
