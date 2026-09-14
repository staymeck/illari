"""Tests for src/report/chart.py — checks the generated HTML contains what's
expected (markers, legend labels), not a visual/pixel snapshot."""
import pandas as pd

from src.report.chart import (
    render_entries_chart,
    render_opportunity_chart,
    write_entries_chart,
    write_opportunity_chart,
)


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


def _real_trades(entry_time, exit_time) -> pd.DataFrame:
    return pd.DataFrame({
        "entry_time": [entry_time], "entry_price": [100.0],
        "exit_time": [exit_time], "exit_price": [102.0],
        "pnl_pct": [1.8], "exit_reason": ["target"],
    })


def _ideal_trades(entry_time, exit_time, direction="long") -> pd.DataFrame:
    return pd.DataFrame({
        "direction": [direction], "entry_time": [entry_time], "entry_price": [100.0],
        "exit_time": [exit_time], "exit_price": [105.0], "mfe_pct": [5.0],
    })


def test_render_opportunity_chart_includes_real_and_ideal_legend_entries():
    df = _candles()
    real = _real_trades(df["timestamp"].iloc[1], df["timestamp"].iloc[3])
    ideal = pd.concat([
        _ideal_trades(df["timestamp"].iloc[0], df["timestamp"].iloc[2], "long"),
        _ideal_trades(df["timestamp"].iloc[1], df["timestamp"].iloc[4], "short"),
    ])

    html = render_opportunity_chart(df, real, ideal, "BTC/USDT", "1h")

    assert "real entry" in html and "real exit" in html
    assert "ideal long entry" in html and "ideal short entry" in html


def test_render_opportunity_chart_handles_empty_inputs():
    df = _candles()

    html = render_opportunity_chart(df, pd.DataFrame(), pd.DataFrame(), "BTC/USDT", "1h")

    assert "real entry" not in html
    assert "ideal long entry" not in html
    assert "BTC" in html


def test_write_opportunity_chart_writes_expected_file(tmp_path):
    df = _candles()
    real = _real_trades(df["timestamp"].iloc[1], df["timestamp"].iloc[3])

    out_path = write_opportunity_chart(df, real, pd.DataFrame(), "BTC/USDT", "1h", reports_dir=tmp_path)

    assert out_path == tmp_path / "btc-usdt_1h_opportunity_vs_reality.html"
    assert out_path.exists()
