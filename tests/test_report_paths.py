"""Tests for src/report/paths.py."""
from pathlib import Path

from src.report.paths import run_report_path, summary_report_path


def test_run_report_path_slugifies_symbol_and_timeframe():
    path = run_report_path("BTC/USDT", "1h", "trend_2024", reports_dir=Path("/tmp/reports"))

    assert path == Path("/tmp/reports/btc-usdt_1h_trend-2024.md")


def test_run_report_path_default_scenario():
    path = run_report_path("ETH/USDT", "5m", reports_dir=Path("/tmp/reports"))

    assert path.name == "eth-usdt_5m_default.md"


def test_summary_report_path():
    path = summary_report_path(reports_dir=Path("/tmp/reports"))

    assert path == Path("/tmp/reports/summary.md")
