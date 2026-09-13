"""Compares multiple catalog strategies across all 5 markets (30m and 1h),
using the cached OHLCV data — validates whether a promising single-market
result holds up broadly, per docs/PLAN.md's rigor discussion (a positive
result on one market with a handful of trades isn't evidence of a real edge
until it's checked elsewhere).

Usage:
    .venv/bin/python scripts/compare_strategies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.backtest.parallel import BacktestJob, run_jobs_in_parallel
from src.report.summary import RunResult, write_summary_report
from src.strategies.builder import load_strategy

SINCE = "2023-09-13"
UNTIL = "2026-09-13"
TIMEFRAMES = ["30m", "1h"]
STRATEGY_PATHS = [
    "config/strategies/trend_pullback_fib.yaml",
    "config/strategies/strong_trend_pullback.yaml",
    "config/strategies/trend_pullback_atr.yaml",
]


def main() -> None:
    strategy_names = [load_strategy(p).name for p in STRATEGY_PATHS]

    jobs = [
        BacktestJob(market.symbol, timeframe, SINCE, UNTIL, scenario=name, strategy_path=path)
        for market in MARKETS
        for timeframe in TIMEFRAMES
        for name, path in zip(strategy_names, STRATEGY_PATHS)
    ]

    print(f"Running {len(jobs)} backtests in parallel...")
    results = run_jobs_in_parallel(jobs)

    market_order = {m.symbol: i for i, m in enumerate(MARKETS)}
    tf_order = {tf: i for i, tf in enumerate(TIMEFRAMES)}
    strat_order = {name: i for i, name in enumerate(strategy_names)}
    results.sort(
        key=lambda r: (market_order[r.job.symbol], tf_order[r.job.timeframe], strat_order[r.job.scenario])
    )

    run_results = []
    for r in results:
        run_results.append(RunResult(r.job.symbol, r.job.timeframe, r.metrics, scenario=r.job.scenario))
        print(
            f"  {r.job.symbol:<10} {r.job.timeframe:<4} {r.job.scenario:<24}: "
            f"{r.metrics['n_trades']:>4} trades, return {r.metrics['total_return_pct']:>7}%"
        )

    out_path = write_summary_report(run_results, reports_dir=Path(__file__).resolve().parents[1] / "reports" / "compare_strategies")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
