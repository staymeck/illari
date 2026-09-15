"""Stage 0 of the 4-stage experiment protocol (Bitácora Illari): freeze the
current best strategy's trade log and summary stats as the fixed baseline
every later stage (+4H filter, +5M confirmation, ...) gets compared
against — one variable at a time, never stacked before being individually
evaluated.

R-multiple based, not pooled equity/return: "risk" per trade is
(entry_price - stop_price)/entry_price*100 and R = pnl_pct / risk_pct. Win
rate, profit factor, expectancy and average winner/loser in R are all
order-independent aggregates, safe to pool across markets and windows —
same reasoning as scripts/run_cost_stress_and_significance.py's pooled
stats. Drawdown is NOT: it's inherently sequence-dependent, so pooling
trades from independent markets into one fake sequence would be exactly
the unsound "pooled total return" this lab already rejected once — it's
computed per market's own chronological trade order instead.
"""
from __future__ import annotations

import pandas as pd

_LOG_COLUMNS = [
    "entry_time",
    "market",
    "tf_execution",
    "direction",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "r_multiple",
    "context_trend",
    "higher_tf_trend",
    "pattern",
    "fib_ratio",
    "volume_bias",
]


def compute_r_multiples(trades: pd.DataFrame) -> pd.Series:
    """Per-trade R: net pnl_pct divided by the risk_pct implied by the
    trade's own entry/stop prices — the strategy-agnostic "how many times
    the risk taken did this trade make or lose"."""
    if trades.empty:
        return pd.Series(dtype=float)
    risk_pct = (trades["entry_price"] - trades["stop_price"]) / trades["entry_price"] * 100
    return trades["pnl_pct"] / risk_pct


def build_trade_log(trades: pd.DataFrame, market: str, tf_execution: str) -> pd.DataFrame:
    """One row per trade with the fields the experiment protocol tracks:
    date, symbol, direction, execution timeframe, entry/SL/TP, result in R,
    and whatever context/pattern extras the strategy's pieces recorded.
    Only columns actually present in `trades` are kept beyond the ones
    added here."""
    if trades.empty:
        return trades

    log = trades.copy()
    log["market"] = market
    log["tf_execution"] = tf_execution
    log["direction"] = "LONG"  # the catalog is long-only throughout this lab
    log["r_multiple"] = compute_r_multiples(trades)

    columns = [c for c in _LOG_COLUMNS if c in log.columns]
    return log[columns].sort_values("entry_time").reset_index(drop=True)


def experiment_report(trades: pd.DataFrame) -> dict:
    """Pooled, order-independent R-multiple stats — safe to compute across
    trades from several markets/windows at once (unlike drawdown, see
    max_drawdown_r_by_market)."""
    if trades.empty:
        return {
            "n_trades": 0, "win_rate_pct": 0.0, "loss_rate_pct": 0.0, "profit_factor": None,
            "expectancy_r": 0.0, "avg_winner_r": 0.0, "avg_loser_r": 0.0,
        }

    # Accepts either raw engine trades (has pnl_pct, R computed fresh) or a
    # build_trade_log() result (already has r_multiple, no pnl_pct column).
    r = trades["r_multiple"] if "r_multiple" in trades.columns else compute_r_multiples(trades)
    wins = r[r > 0]
    losses = r[r <= 0]
    n = len(r)
    gross_win = wins.sum()
    gross_loss = -losses.sum()

    return {
        "n_trades": n,
        "win_rate_pct": round(100 * len(wins) / n, 2),
        "loss_rate_pct": round(100 * len(losses) / n, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy_r": round(r.mean(), 3),
        "avg_winner_r": round(wins.mean(), 3) if not wins.empty else 0.0,
        "avg_loser_r": round(losses.mean(), 3) if not losses.empty else 0.0,
    }


def max_drawdown_r_by_market(trades: pd.DataFrame) -> pd.DataFrame:
    """Max drawdown of the CUMULATIVE R curve, computed separately per
    market in that market's own chronological trade order (never pooled —
    there's no single account that took trades from 5 independent markets
    in one sequence). One row per market: n_trades, max_drawdown_r.

    Accepts either raw engine trades (has pnl_pct, R computed fresh here)
    or a build_trade_log() result (already has r_multiple — used as-is,
    since pnl_pct isn't part of the log's column set)."""
    if trades.empty or "market" not in trades.columns:
        return pd.DataFrame()

    rows = []
    for market, group in trades.groupby("market"):
        group = group.sort_values("entry_time")
        r = group["r_multiple"] if "r_multiple" in group.columns else compute_r_multiples(group)
        cum_r = r.cumsum()
        running_max = cum_r.cummax()
        drawdown = cum_r - running_max
        rows.append({"market": market, "n_trades": len(group), "max_drawdown_r": round(drawdown.min(), 3)})
    return pd.DataFrame(rows)
