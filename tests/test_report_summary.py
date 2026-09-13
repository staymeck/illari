"""Tests for src/report/summary.py."""
from src.report.summary import RunResult, render_summary_markdown, write_summary_report

_METRICS_A = {"n_trades": 3, "win_rate_pct": 33.3, "profit_factor": 0.5, "total_return_pct": -1.5}
_METRICS_B = {"n_trades": 20, "win_rate_pct": 55.0, "profit_factor": 1.8, "total_return_pct": 12.0}


def test_render_summary_markdown_empty():
    assert "No runs yet" in render_summary_markdown([])


def test_render_summary_markdown_lists_all_runs_and_links():
    results = [
        RunResult("BTC/USDT", "1h", _METRICS_A, scenario="minimal_validation"),
        RunResult("ETH/USDT", "1d", _METRICS_B, scenario="trend_2024"),
    ]

    markdown = render_summary_markdown(results)

    assert "BTC/USDT" in markdown
    assert "ETH/USDT" in markdown
    assert "btc-usdt_1h_minimal-validation.md" in markdown
    assert "eth-usdt_1d_trend-2024.md" in markdown


def test_write_summary_report_writes_to_standard_path(tmp_path):
    results = [RunResult("BTC/USDT", "1h", _METRICS_A)]

    out_path = write_summary_report(results, reports_dir=tmp_path)

    assert out_path == tmp_path / "summary.md"
    assert out_path.exists()
    assert "BTC/USDT" in out_path.read_text(encoding="utf-8")
