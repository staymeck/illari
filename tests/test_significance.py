"""Tests for src/backtest/significance.py."""
import pandas as pd

from src.backtest.significance import breakeven_win_rate, significance_report


def test_breakeven_win_rate_known_value():
    # avg win 200, avg loss 100 -> breakeven rate = 100/(200+100) = 33.33%
    trades = pd.DataFrame({"pnl_abs": [200.0, 200.0, -100.0, -100.0, -100.0]})

    assert round(breakeven_win_rate(trades) * 100, 2) == 33.33


def test_breakeven_win_rate_defaults_to_half_with_no_wins_or_losses():
    assert breakeven_win_rate(pd.DataFrame({"pnl_abs": []})) == 0.5


def test_significance_report_empty_trades():
    report = significance_report(pd.DataFrame({"pnl_abs": []}))

    assert report["n_trades"] == 0
    assert report["breakeven_inside_ci"] is True
    assert report["p_value_vs_breakeven"] is None


def test_significance_report_flags_breakeven_inside_ci_for_a_small_sample():
    # Only 5 trades: even a 60% observed win rate can't rule out a lower
    # true rate with any confidence - breakeven should fall inside the CI.
    trades = pd.DataFrame({"pnl_abs": [200.0, 200.0, 200.0, -100.0, -100.0]})

    report = significance_report(trades)

    assert report["n_trades"] == 5
    assert report["breakeven_inside_ci"] is True


def test_significance_report_rejects_breakeven_for_a_large_clearly_winning_sample():
    # 200 trades, 70% win rate, 2:1 win/loss size -> breakeven ~33%, and
    # with this much data the 95% CI should sit well clear of it.
    pnl = [200.0] * 140 + [-100.0] * 60
    trades = pd.DataFrame({"pnl_abs": pnl})

    report = significance_report(trades)

    assert report["breakeven_inside_ci"] is False
    assert report["p_value_vs_breakeven"] < 0.05
