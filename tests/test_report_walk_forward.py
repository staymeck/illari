"""Tests for src/report/walk_forward.py."""
from src.backtest.walk_forward import Fold, FoldResult
from src.report.walk_forward import render_walk_forward_markdown, write_walk_forward_report

_FOLD_RESULTS = [
    FoldResult(Fold("2023-01-01", "2024-01-01"), {"n_trades": 10, "win_rate_pct": 40.0, "profit_factor": 0.8, "total_return_pct": -2.0}),
    FoldResult(Fold("2024-01-01", "2025-01-01"), {"n_trades": 8, "win_rate_pct": 60.0, "profit_factor": 1.5, "total_return_pct": 3.0}),
    FoldResult(Fold("2025-01-01", "2026-01-01"), {"n_trades": 12, "win_rate_pct": 55.0, "profit_factor": 1.2, "total_return_pct": 1.5}),
]


def test_render_walk_forward_markdown_lists_every_fold_and_counts_positives():
    markdown = render_walk_forward_markdown("BTC/USDT", "1h", "strong_trend_pullback", _FOLD_RESULTS)

    assert "BTC/USDT" in markdown
    assert "strong_trend_pullback" in markdown
    assert "2023-01-01" in markdown and "2026-01-01" in markdown
    assert "2/3 folds positive" in markdown


def test_write_walk_forward_report_writes_expected_file(tmp_path):
    out_path = write_walk_forward_report("BTC/USDT", "1h", "strong_trend_pullback", _FOLD_RESULTS, reports_dir=tmp_path)

    assert out_path == tmp_path / "walk_forward" / "btc-usdt_1h_strong_trend_pullback.md"
    assert out_path.exists()
    assert "BTC/USDT" in out_path.read_text(encoding="utf-8")
