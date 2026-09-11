import pandas as pd
import pytest

from trading_lab.backtest import metrics
from trading_lab.backtest.engine import BacktestResult, Trade


def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=i)


def _trade(pnl: float, hour_utc: int, session: str) -> Trade:
    return Trade(
        entry_time=_ts(0),
        exit_time=_ts(1),
        direction="long",
        entry_price=100.0,
        exit_price=100.0 + pnl,
        stop_loss=90.0,
        take_profit=120.0,
        position_size=1.0,
        pnl=pnl,
        pnl_pct=pnl / 1000.0 * 100.0,
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        confidence_score=4,
        hour_utc=hour_utc,
        session=session,
        reasons={},
    )


def _trades() -> list[Trade]:
    return [
        _trade(100.0, hour_utc=1, session="asia"),
        _trade(-40.0, hour_utc=1, session="asia"),
        _trade(50.0, hour_utc=2, session="london"),
        _trade(-60.0, hour_utc=2, session="london"),
    ]


def test_win_rate():
    df = metrics.trades_to_df(_trades())
    assert metrics.win_rate(df) == pytest.approx(50.0)


def test_profit_factor():
    df = metrics.trades_to_df(_trades())
    # gross win = 100+50=150, gross loss = 40+60=100 -> 1.5
    assert metrics.profit_factor(df) == pytest.approx(1.5)


def test_profit_factor_no_losses_is_infinite():
    df = metrics.trades_to_df([_trade(100.0, 1, "asia"), _trade(50.0, 2, "london")])
    assert metrics.profit_factor(df) == float("inf")


def test_profit_factor_empty_is_zero():
    df = metrics.trades_to_df([])
    assert metrics.profit_factor(df) == 0.0


def test_expectancy():
    df = metrics.trades_to_df(_trades())
    # (100 - 40 + 50 - 60) / 4 = 12.5
    assert metrics.expectancy(df) == pytest.approx(12.5)


def test_max_drawdown_pct():
    equity_curve = pd.DataFrame(
        {
            "timestamp": [_ts(i) for i in range(5)],
            "equity": [1000.0, 1100.0, 1060.0, 1110.0, 1050.0],
        }
    )
    dd = metrics.max_drawdown_pct(equity_curve)
    # Worst drop: from 1110 (running max) to 1050 -> -5.405...%
    assert dd == pytest.approx((1050.0 - 1110.0) / 1110.0 * 100.0)


def test_total_return_pct():
    equity_curve = pd.DataFrame({"timestamp": [_ts(0), _ts(1)], "equity": [1000.0, 1050.0]})
    assert metrics.total_return_pct(equity_curve, initial_capital=1000.0) == pytest.approx(5.0)


def test_breakdown_by_hour_utc():
    result = BacktestResult(
        trades=_trades(),
        equity_curve=pd.DataFrame({"timestamp": [_ts(0)], "equity": [1050.0]}),
        final_equity=1050.0,
    )
    breakdown = metrics.breakdown_by(result, "hour_utc")

    row_h1 = breakdown[breakdown["hour_utc"] == 1].iloc[0]
    assert row_h1["n_trades"] == 2
    assert row_h1["win_rate_pct"] == pytest.approx(50.0)
    assert row_h1["profit_factor"] == pytest.approx(100.0 / 40.0)
    assert row_h1["avg_pnl"] == pytest.approx((100.0 - 40.0) / 2)

    row_h2 = breakdown[breakdown["hour_utc"] == 2].iloc[0]
    assert row_h2["profit_factor"] == pytest.approx(50.0 / 60.0, abs=0.01)  # the table rounds to 2 decimals


def test_summarize_matches_individual_metrics():
    equity_curve = pd.DataFrame(
        {"timestamp": [_ts(i) for i in range(5)], "equity": [1000.0, 1100.0, 1060.0, 1110.0, 1050.0]}
    )
    result = BacktestResult(trades=_trades(), equity_curve=equity_curve, final_equity=1050.0)
    summary = metrics.summarize(result, initial_capital=1000.0)

    assert summary["n_trades"] == 4
    assert summary["win_rate_pct"] == pytest.approx(50.0)
    assert summary["total_return_pct"] == pytest.approx(5.0)
