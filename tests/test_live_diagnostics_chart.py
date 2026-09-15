"""Tests for src/report/live_diagnostics_chart.py — checks the returned
HTML contains the expected embedded data/markup, not a visual snapshot
(same convention as tests/test_chart.py)."""
import json

import pandas as pd

from src.report.live_diagnostics_chart import render_live_diagnostics_chart, write_live_diagnostics_chart


def _df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "open": [100, 101, 102], "high": [102, 103, 104], "low": [99, 100, 101], "close": [101, 102, 103]}
    )


def test_render_includes_symbol_and_candlestick_data():
    html = render_live_diagnostics_chart(_df(), {}, "BTC/USDT")

    assert "BTC/USDT" in html
    assert "Candlestick" in html or "candlestick" in html.lower()
    assert "plotly_click" in html


def test_render_embeds_the_diagnostics_dict_as_json():
    ts = "2024-01-01 00:00:00+00:00"
    diagnostics = {
        ts: {
            "context_value": "uptrend",
            "context_required": "uptrend",
            "context_passed": True,
            "setup_passed": True,
            "confirmations": [{"piece": "volume", "passed": False}],
            "signal": False,
        }
    }

    html = render_live_diagnostics_chart(_df(), diagnostics, "BTC/USDT")

    assert json.dumps(diagnostics) in html
    assert "SEÑAL FINAL" in html


def test_render_x_axis_timestamps_match_diagnose_entry_key_format():
    # Regression: diagnose_entry keys traces by str(pd.Timestamp) ("...
    # 00:00:00+00:00"), so the candlestick's own x values must be
    # serialized the exact same way, or the click handler's dict lookup
    # never matches anything (plotly's default Timestamp->JSON formatting
    # drops the tz offset and uses "T" instead of a space).
    df = _df()
    expected_key = str(df["timestamp"].iloc[0])  # "...+00:00"

    html = render_live_diagnostics_chart(df, {}, "BTC/USDT")

    assert expected_key in html
    assert "T00:00:00\"" not in html  # the mismatched plotly-native format must not appear


def test_write_live_diagnostics_chart_creates_file(tmp_path):
    output_path = tmp_path / "sub" / "chart.html"

    result = write_live_diagnostics_chart(_df(), {}, "ETH/USDT", output_path)

    assert result == output_path
    assert output_path.exists()
    assert "ETH/USDT" in output_path.read_text(encoding="utf-8")
