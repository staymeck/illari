"""Tests for src/report/render.py."""
from pathlib import Path

import pandas as pd

from src.report.render import _stop_noise_lines, render_run_markdown, write_run_report

_METRICS = {
    "n_trades": 2,
    "win_rate_pct": 50.0,
    "profit_factor": 1.5,
    "expectancy": 10.0,
    "max_drawdown_pct": -5.0,
    "total_return_pct": 2.0,
    "final_equity": 10200.0,
}


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_utc": [9, 14],
            "session": ["london", "new_york"],
            "pattern": ["hammer", "bullish_engulfing"],
            "exit_reason": ["target", "stop"],
            "stop_was_premature": [None, True],
            "pnl_abs": [150.0, -50.0],
        }
    )


def test_render_run_markdown_includes_market_and_metrics():
    markdown = render_run_markdown("BTC/USDT", "1h", _trades(), _METRICS, scenario="trend_2024")

    assert "BTC/USDT" in markdown
    assert "1h" in markdown
    assert "trend_2024" in markdown
    assert "**2**" in markdown  # n_trades
    assert "Breakdown by exit reason" in markdown
    assert "Breakdown by entry hour" in markdown
    assert "Breakdown by market session" in markdown
    assert "Breakdown by confirming candlestick pattern" in markdown
    assert "Stop-loss noise diagnostic" in markdown
    assert "1 (100.0%)" in markdown  # the one stop-out was premature


def test_render_run_markdown_links_chart_when_given():
    markdown = render_run_markdown(
        "BTC/USDT", "1h", _trades(), _METRICS, chart_path=Path("/anywhere/btc-usdt_1h_entries_chart.html")
    )

    assert "btc-usdt_1h_entries_chart.html" in markdown


def test_render_run_markdown_no_chart_link_by_default():
    markdown = render_run_markdown("BTC/USDT", "1h", _trades(), _METRICS)

    assert "entries_chart" not in markdown


def test_render_run_markdown_handles_no_trades():
    markdown = render_run_markdown("BTC/USDT", "1h", pd.DataFrame(), _METRICS)

    assert "No trades were generated" in markdown


def test_stop_noise_lines_empty_when_no_stop_exits():
    trades = pd.DataFrame({"exit_reason": ["target", "timeout"], "stop_was_premature": [None, None]})

    assert _stop_noise_lines(trades) == []


def test_stop_noise_lines_reports_correct_percentage():
    trades = pd.DataFrame(
        {
            "exit_reason": ["stop", "stop", "stop", "target"],
            "stop_was_premature": [True, True, False, None],
        }
    )

    lines = _stop_noise_lines(trades)
    text = "\n".join(lines)

    assert "Stopped-out trades: **3**" in text
    assert "2 (66.67%)" in text


def test_write_run_report_writes_to_standard_path(tmp_path):
    out_path = write_run_report("BTC/USDT", "1h", _trades(), _METRICS, scenario="trend_2024", reports_dir=tmp_path)

    assert out_path == tmp_path / "btc-usdt_1h_trend-2024.md"
    assert out_path.exists()
    assert "BTC/USDT" in out_path.read_text(encoding="utf-8")
