"""Backtest metrics: overall, and broken down by hour/session."""

from __future__ import annotations

import pandas as pd

from trading_lab.backtest.engine import BacktestResult, Trade


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time", "exit_time", "direction", "entry_price", "exit_price",
                "stop_loss", "take_profit", "position_size", "pnl", "pnl_pct",
                "exit_reason", "confidence_score", "hour_utc", "session",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])


def win_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["pnl"] > 0).mean() * 100.0)


def profit_factor(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    gross_win = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = -df.loc[df["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def expectancy(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["pnl"].mean())


def max_drawdown_pct(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    equity = equity_curve["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100.0
    return float(drawdown.min())


def total_return_pct(equity_curve: pd.DataFrame, initial_capital: float) -> float:
    if equity_curve.empty:
        return 0.0
    final = equity_curve["equity"].iloc[-1]
    return float((final - initial_capital) / initial_capital * 100.0)


def summarize(result: BacktestResult, initial_capital: float) -> dict:
    df = trades_to_df(result.trades)
    return {
        "n_trades": int(len(df)),
        "win_rate_pct": round(win_rate(df), 2),
        "profit_factor": round(profit_factor(df), 2) if profit_factor(df) != float("inf") else float("inf"),
        "expectancy": round(expectancy(df), 2),
        "max_drawdown_pct": round(max_drawdown_pct(result.equity_curve), 2),
        "total_return_pct": round(total_return_pct(result.equity_curve, initial_capital), 2),
        "final_equity": round(result.final_equity, 2),
    }


def breakdown_by(result: BacktestResult, column: str) -> pd.DataFrame:
    """Breaks down win rate / profit factor / trade count / average pnl by
    `column` (e.g. 'hour_utc' or 'session')."""
    df = trades_to_df(result.trades)
    if df.empty:
        return pd.DataFrame(columns=[column, "n_trades", "win_rate_pct", "profit_factor", "avg_pnl"])

    rows = []
    for key, group in df.groupby(column):
        rows.append(
            {
                column: key,
                "n_trades": int(len(group)),
                "win_rate_pct": round(win_rate(group), 2),
                "profit_factor": round(profit_factor(group), 2) if profit_factor(group) != float("inf") else float("inf"),
                "avg_pnl": round(group["pnl"].mean(), 2),
            }
        )
    result_df = pd.DataFrame(rows)
    sort_col = column if column == "hour_utc" else "n_trades"
    ascending = column == "hour_utc"
    return result_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
