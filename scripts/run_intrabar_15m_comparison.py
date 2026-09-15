"""Re-runs the intrabar entry-timing test the project already did once at
5-minute resolution (rejected: avg return +3.03% -> -6.10%, more than
double the trade count, worse exit mix — see docs/PLAN.md / Bitácora
Illari and the "Illari Control Room" artifact's own journey) — this time
at 15-minute resolution specifically, against the CURRENT live strategy
(trend_pullback_htf_confluence.yaml, not the older structural_stop-only
config the original 5m test used), because the user's specific question
is about 15-minute checks, not 5-minute ones, and the two aren't
necessarily identical - don't assume, measure.

Compares run_backtest (standard, 1h close-based) against
run_intrabar_backtest (checking every 15 min as the bar forms) on all 5
markets, both known and virgin windows, plus the same exit-reason/
stop-premature breakdown that explained WHY 5-minute checking failed
(more stop-outs on ordinary noise, not better entries) — to see whether
the same mechanism applies at 15-minute granularity or whether a coarser
check avoids it.

Usage:
    .venv/bin/python scripts/run_intrabar_15m_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import compute_metrics, run_backtest
from src.backtest.experiment import build_trade_log, experiment_report
from src.backtest.intrabar_entry import run_intrabar_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

PARENT_TIMEFRAME = "1h"
CHILD_TIMEFRAME = "15m"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-15"), ("virgin", "2020-09-01", "2023-09-13")]

STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    std_logs, intra_logs = [], []

    for label, since, until in WINDOWS:
        print(f"\n=== {label}: {since} -> {until} ===\n")
        print(f"{'market':<10}{'std_n':>7}{'std_ret%':>10}{'intra_n':>9}{'intra_ret%':>12}")
        for market in MARKETS:
            parent_df = fetch_ohlcv(market.symbol, PARENT_TIMEFRAME, since=since, until=until)
            child_df = fetch_ohlcv(market.symbol, CHILD_TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)

            std_trades, std_equity = run_backtest(parent_df, strategy, higher_tf_df=higher_tf_df)
            std_metrics = compute_metrics(std_trades, std_equity, strategy.initial_equity)

            intra_trades, intra_equity = run_intrabar_backtest(parent_df, child_df, strategy, higher_tf_df=higher_tf_df)
            intra_metrics = compute_metrics(intra_trades, intra_equity, strategy.initial_equity)

            print(f"{market.symbol:<10}{std_metrics['n_trades']:>7}{std_metrics['total_return_pct']:>10.2f}"
                  f"{intra_metrics['n_trades']:>9}{intra_metrics['total_return_pct']:>12.2f}")

            if not std_trades.empty:
                std_logs.append(build_trade_log(std_trades, market=market.symbol, tf_execution="1h"))
            if not intra_trades.empty:
                intra_logs.append(build_trade_log(intra_trades, market=market.symbol, tf_execution="15m"))

    std_pool = pd.concat(std_logs, ignore_index=True) if std_logs else pd.DataFrame()
    intra_pool = pd.concat(intra_logs, ignore_index=True) if intra_logs else pd.DataFrame()

    print("\n=== POOLED (both windows, all 5 markets) ===\n")
    for label, pool in [("standard (1h close)", std_pool), ("intrabar (15min checks)", intra_pool)]:
        report = experiment_report(pool)
        pf = report["profit_factor"]
        pf_str = f"{pf:.2f}" if pf is not None else "n/a"
        print(f"{label:<26} n={report['n_trades']:<5} win_rate={report['win_rate_pct']:.2f}%  "
              f"PF={pf_str}  expectancy={report['expectancy_r']:.3f}R")
        if not pool.empty and "exit_reason" in pool.columns:
            breakdown = pool["exit_reason"].value_counts(normalize=True) * 100
            print(f"  exit_reason mix: {breakdown.round(1).to_dict()}")


if __name__ == "__main__":
    main()
