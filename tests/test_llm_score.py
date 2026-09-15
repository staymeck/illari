"""Tests for src/analysis/llm_score.py — offline only, per the project's
convention (a live Ollama server is a manual-script concern, not something
the suite depends on). The HTTP call is mocked."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from src.analysis.llm_score import format_candles, format_context, parse_score, score_candles


def _window(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [100.6 + i for i in range(n)],
            "low": [99.8 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
        }
    )


def test_format_candles_lists_ohlc_oldest_to_newest():
    text = format_candles(_window())

    lines = text.splitlines()
    assert len(lines) == 2
    assert "O=100.0000" in lines[0] and "C=100.5000" in lines[0]
    assert "O=101.0000" in lines[1] and "C=101.5000" in lines[1]


def test_format_context_lists_only_available_fields():
    text = format_context({"pattern": "hammer", "context_trend": "uptrend", "fib_ratio": None})

    assert "hammer" in text
    assert "uptrend" in text
    assert "Fibonacci" not in text  # fib_ratio was None -> excluded


def test_format_context_placeholder_when_nothing_available():
    assert format_context(None) == "(no additional analysis fields available)"
    assert format_context({}) == "(no additional analysis fields available)"


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
        score = score_candles(_window(), context={"pattern": "hammer"})

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


def test_score_candles_sends_the_requested_num_ctx():
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"response": "50"}).encode()
    fake_response.__enter__.return_value = fake_response

    with patch("src.analysis.llm_score.urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
        score_candles(_window(), num_ctx=8192)

    sent_request = mock_urlopen.call_args[0][0]
    sent_payload = json.loads(sent_request.data)
    assert sent_payload["options"]["num_ctx"] == 8192
