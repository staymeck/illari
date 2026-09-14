"""Tests for src/backtest/experiment.py."""
import pandas as pd

from src.backtest.experiment import (
    build_trade_log,
    compute_r_multiples,
    experiment_report,
    max_drawdown_r_by_market,
)


def _trades() -> pd.DataFrame:
    # risk_pct = (entry-stop)/entry*100 = 2.0% for every trade here, so R is
    # easy to hand-check: pnl_pct / 2.0.
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"]),
            "exit_time": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-06", "2024-01-08"]),
            "entry_price": [100.0, 100.0, 100.0, 100.0],
            "stop_price": [98.0, 98.0, 98.0, 98.0],
            "target_price": [104.0, 104.0, 104.0, 104.0],
            "exit_price": [104.0, 98.0, 102.0, 98.0],
            "exit_reason": ["target", "stop", "timeout", "stop"],
            "pnl_pct": [4.0, -2.0, 1.0, -2.0],  # -> R: 2.0, -1.0, 0.5, -1.0
            "context_trend": ["uptrend"] * 4,
        }
    )


def test_compute_r_multiples_known_values():
    r = compute_r_multiples(_trades())

    assert list(r.round(3)) == [2.0, -1.0, 0.5, -1.0]


def test_compute_r_multiples_empty():
    assert compute_r_multiples(pd.DataFrame()).empty


def test_build_trade_log_adds_expected_columns_and_sorts_chronologically():
    log = build_trade_log(_trades(), market="BTC/USDT", tf_execution="1h")

    assert list(log["market"].unique()) == ["BTC/USDT"]
    assert list(log["direction"].unique()) == ["LONG"]
    assert list(log["tf_execution"].unique()) == ["1h"]
    assert list(log["entry_time"]) == sorted(log["entry_time"])
    assert "r_multiple" in log.columns
    assert "unknown_column" not in log.columns


def test_build_trade_log_empty_input():
    assert build_trade_log(pd.DataFrame(), market="BTC/USDT", tf_execution="1h").empty


def test_experiment_report_known_values():
    report = experiment_report(_trades())

    assert report["n_trades"] == 4
    assert report["win_rate_pct"] == 50.0
    assert report["loss_rate_pct"] == 50.0
    assert report["expectancy_r"] == round((2.0 - 1.0 + 0.5 - 1.0) / 4, 3)
    assert report["avg_winner_r"] == round((2.0 + 0.5) / 2, 3)
    assert report["avg_loser_r"] == -1.0
    assert report["profit_factor"] == round((2.0 + 0.5) / (1.0 + 1.0), 2)


def test_experiment_report_empty():
    report = experiment_report(pd.DataFrame())

    assert report["n_trades"] == 0
    assert report["profit_factor"] is None


def test_max_drawdown_r_by_market_known_sequence():
    # Market A: R sequence +2, -1, +0.5, -1 -> cumulative 2, 1, 1.5, 0.5.
    # Running max hits 2 after the first trade, so the worst drawdown is
    # 0.5 - 2 = -1.5 (after the last trade).
    trades = build_trade_log(_trades(), market="A", tf_execution="1h")

    result = max_drawdown_r_by_market(trades)
    row = result[result["market"] == "A"].iloc[0]

    assert row["n_trades"] == 4
    assert row["max_drawdown_r"] == -1.5


def test_max_drawdown_r_by_market_is_separate_per_market():
    log_a = build_trade_log(_trades(), market="A", tf_execution="1h")
    log_b = build_trade_log(_trades(), market="B", tf_execution="1h")
    combined = pd.concat([log_a, log_b], ignore_index=True)

    result = max_drawdown_r_by_market(combined)

    assert sorted(result["market"]) == ["A", "B"]
    assert result[result["market"] == "A"]["max_drawdown_r"].iloc[0] == -1.5
    assert result[result["market"] == "B"]["max_drawdown_r"].iloc[0] == -1.5


def test_max_drawdown_r_by_market_empty():
    assert max_drawdown_r_by_market(pd.DataFrame()).empty
