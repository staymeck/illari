"""Tests on synthetic data for src/backtest/engine.py."""
import pandas as pd

from src.backtest.engine import compute_metrics


def test_compute_metrics_known_trades():
    # 2 winners (+100 each) and 1 loser (-50): result known by hand.
    trades = pd.DataFrame(
        {
            "pnl_abs": [100.0, 100.0, -50.0],
            "equity_after": [10100.0, 10200.0, 10150.0],
        }
    )
    equity_curve = pd.Series([10000.0, 10100.0, 10200.0, 10150.0])

    metrics = compute_metrics(trades, equity_curve, initial_equity=10000.0)

    assert metrics["n_trades"] == 3
    assert metrics["win_rate_pct"] == round(2 / 3 * 100, 2)
    assert metrics["profit_factor"] == round(200 / 50, 2)
    assert metrics["expectancy"] == round((100 + 100 - 50) / 3, 2)
    assert metrics["final_equity"] == 10150.0
    assert metrics["total_return_pct"] == 1.5
    # drawdown from the 10200 peak down to 10150 = -0.49%
    assert metrics["max_drawdown_pct"] == round((10150 - 10200) / 10200 * 100, 2)


def test_compute_metrics_no_trades_returns_zeros():
    metrics = compute_metrics(pd.DataFrame(), pd.Series([10000.0]), initial_equity=10000.0)

    assert metrics["n_trades"] == 0
    assert metrics["final_equity"] == 10000.0
