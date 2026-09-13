"""Compares scheduled_3am_6pm.yaml (enter every day at a fixed hour, exit at
a fixed hour, no market condition at all) against the random-entry Monte
Carlo benchmark with the same trade count — does a specific time-of-day
window carry a consistent bias, or does it perform like any other random
timing? See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/run_scheduled_vs_random.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import compute_metrics, run_backtest
from src.backtest.random_benchmark import percentile_rank, run_monte_carlo
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"
N_SIMULATIONS = 500
SEED = 42


def main() -> None:
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df)} candles\n")

    strategy = load_strategy("config/strategies/scheduled_3am_6pm.yaml")
    trades, equity = run_backtest(df, strategy)
    metrics = compute_metrics(trades, equity, strategy.initial_equity)
    print(f"scheduled_3am_6pm: {metrics}")

    mc = run_monte_carlo(df, strategy, target_n_trades=metrics["n_trades"], n_simulations=N_SIMULATIONS, seed=SEED)
    print(f"\nRandom benchmark ({N_SIMULATIONS} sims, ~{metrics['n_trades']} trades each):")
    print(mc[["n_trades", "total_return_pct", "win_rate_pct", "profit_factor"]].describe())

    rank = percentile_rank(metrics["total_return_pct"], mc["total_return_pct"])
    print(f"\nscheduled_3am_6pm total_return_pct percentile vs random: {rank:.1f}")


if __name__ == "__main__":
    main()
