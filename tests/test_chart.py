"""Tests for src/report/chart.py — checks the generated HTML contains what's
expected (markers, legend labels), not a visual/pixel snapshot."""
import pandas as pd

from src.report.chart import render_entries_chart, write_entries_chart


def _candles(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
        }
    )


def _trades(entry_time, entry_price=100.0) -> pd.DataFrame:
    return pd.DataFrame({"entry_time": [entry_time], "entry_price": [entry_price]})


def test_render_entries_chart_includes_symbol_and_every_kind_present():
    df = _candles()
    trades_by_kind = {
        "deterministic": _trades(df["timestamp"].iloc[1]),
        "probabilistic": _trades(df["timestamp"].iloc[2]),
        "random": _trades(df["timestamp"].iloc[3]),
    }

    html = render_entries_chart(df, trades_by_kind, "BTC/USDT", "1h")

    assert "BTC" in html and "USDT" in html
    assert "deterministic entry" in html
    assert "probabilistic entry" in html
    assert "random entry" in html


def test_render_entries_chart_skips_empty_or_missing_kinds():
    df = _candles()
    trades_by_kind = {
        "deterministic": _trades(df["timestamp"].iloc[1]),
        "probabilistic": pd.DataFrame(),  # empty -> no trace
    }

    html = render_entries_chart(df, trades_by_kind, "BTC/USDT", "1h")

    assert "deterministic entry" in html
    assert "probabilistic entry" not in html
    assert "random entry" not in html


def test_write_entries_chart_writes_expected_file(tmp_path):
    df = _candles()
    trades_by_kind = {"deterministic": _trades(df["timestamp"].iloc[1])}

    out_path = write_entries_chart(df, trades_by_kind, "BTC/USDT", "1h", reports_dir=tmp_path)

    assert out_path == tmp_path / "btc-usdt_1h_entries_chart.html"
    assert out_path.exists()
    assert "BTC" in out_path.read_text(encoding="utf-8")
