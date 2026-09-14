"""Tests for src/report/trade_audit.py."""
import pandas as pd

from src.report.trade_audit import build_trade_audit, pattern_breakdown, render_trade_audit_markdown


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp("2024-01-02"), "context_trend": "uptrend",
                "pattern": "hammer", "entry_price": 100.0, "stop_price": 98.0,
                "target_price": 104.0, "exit_time": pd.Timestamp("2024-01-03"),
                "exit_price": 104.0, "exit_reason": "target", "stop_was_premature": None,
                "pnl_pct": 3.8, "pnl_abs": 380.0,
            },
            {
                "entry_time": pd.Timestamp("2024-01-01"), "context_trend": "uptrend",
                "pattern": "hammer", "entry_price": 90.0, "stop_price": 88.0,
                "target_price": 94.0, "exit_time": pd.Timestamp("2024-01-01T12:00"),
                "exit_price": 88.0, "exit_reason": "stop", "stop_was_premature": True,
                "pnl_pct": -2.2, "pnl_abs": -220.0,
            },
            {
                "entry_time": pd.Timestamp("2024-01-05"), "context_trend": "uptrend",
                "pattern": "morning_star", "entry_price": 110.0, "stop_price": 108.0,
                "target_price": 114.0, "exit_time": pd.Timestamp("2024-01-06"),
                "exit_price": 108.0, "exit_reason": "stop", "stop_was_premature": False,
                "pnl_pct": -2.0, "pnl_abs": -200.0,
            },
        ]
    )


def test_build_trade_audit_sorts_chronologically_and_keeps_known_columns():
    audit = build_trade_audit(_trades())

    assert list(audit["entry_time"]) == sorted(_trades()["entry_time"])
    assert "pattern" in audit.columns
    assert "unknown_column" not in audit.columns


def test_build_trade_audit_empty_input():
    result = build_trade_audit(pd.DataFrame())

    assert result.empty


def test_render_trade_audit_markdown_includes_a_row_per_trade():
    markdown = render_trade_audit_markdown(_trades(), title="BTC/USDT — known window")

    assert "BTC/USDT" in markdown
    assert markdown.count("hammer") == 2
    assert "morning_star" in markdown
    assert "target" in markdown and "stop" in markdown


def test_render_trade_audit_markdown_handles_no_trades():
    markdown = render_trade_audit_markdown(pd.DataFrame(), title="Empty case")

    assert "no trades" in markdown


def test_pattern_breakdown_separates_patterns_and_computes_win_rate():
    breakdown = pattern_breakdown(_trades())

    hammer_row = breakdown[breakdown["pattern"] == "hammer"].iloc[0]
    assert hammer_row["n_trades"] == 2
    assert hammer_row["win_rate_pct"] == 50.0

    star_row = breakdown[breakdown["pattern"] == "morning_star"].iloc[0]
    assert star_row["n_trades"] == 1
    assert star_row["win_rate_pct"] == 0.0
    assert star_row["stop_pct"] == 100.0


def test_pattern_breakdown_empty_without_pattern_column():
    trades = pd.DataFrame({"pnl_abs": [1.0, -1.0]})

    assert pattern_breakdown(trades).empty
