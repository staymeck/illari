"""Phase 1 full scale-up: 5 markets x 3 timeframes x 36 months (docs/PLAN.md).

Two phases, deliberately kept separate:

1. Sequentially warm the on-disk OHLCV cache for every (market, timeframe)
   combination. This is the only phase that hits Binance's API, and running
   it sequentially (one process, ccxt's own rate limiting) avoids 15 worker
   processes independently hammering the API at once and tripping Binance's
   per-IP rate limit.
2. Run all 15 backtests in parallel (src/backtest/parallel.py) — by then
   every fetch_ohlcv() call inside a worker hits the local parquet cache, so
   this phase is pure CPU with no network contention.

Usage:
    .venv/bin/python scripts/run_full_scan.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS, TIMEFRAMES
from src.backtest.parallel import BacktestJob, run_jobs_in_parallel
from src.data.fetcher import earliest_available, fetch_ohlcv
from src.report.render import write_run_report
from src.report.summary import RunResult, write_summary_report

SCENARIO = "last_36_months"
MONTHS = 36


def _scenario_window(symbol: str) -> tuple[str, str]:
    """Adjusts the start date to the market's real listing date when it
    wasn't trading yet 36 months ago (see docs/PLAN.md listing-date caveat)."""
    until = pd.Timestamp.now(tz="UTC").normalize()
    target_since = until - pd.DateOffset(months=MONTHS)
    listed_since = earliest_available(symbol, timeframe="1d")
    since = max(target_since, listed_since + pd.Timedelta(days=1))
    return since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")


def main() -> None:
    jobs: list[BacktestJob] = []
    for market in MARKETS:
        since, until = _scenario_window(market.symbol)
        for timeframe in TIMEFRAMES:
            jobs.append(BacktestJob(market.symbol, timeframe, since, until, scenario=SCENARIO))

    print(f"Phase 1/2: warming the OHLCV cache for {len(jobs)} (market, timeframe) combinations...")
    t0 = time.time()
    for job in jobs:
        df = fetch_ohlcv(job.symbol, job.timeframe, since=job.since, until=job.until)
        print(f"  {job.symbol:<10} {job.timeframe:<3} {job.since} -> {job.until}: {len(df)} candles")
    print(f"Cache warmed in {time.time() - t0:.1f}s\n")

    print("Phase 2/2: running all backtests in parallel...")
    t0 = time.time()
    results = run_jobs_in_parallel(jobs)
    print(f"Ran {len(results)} backtests in {time.time() - t0:.1f}s\n")

    order = {m.symbol: i for i, m in enumerate(MARKETS)}
    tf_order = {tf: i for i, tf in enumerate(TIMEFRAMES)}
    results.sort(key=lambda r: (order[r.job.symbol], tf_order[r.job.timeframe]))

    run_results = []
    for r in results:
        out_path = write_run_report(r.job.symbol, r.job.timeframe, r.trades, r.metrics, scenario=r.job.scenario)
        run_results.append(RunResult(r.job.symbol, r.job.timeframe, r.metrics, scenario=r.job.scenario))
        print(f"  {r.job.symbol:<10} {r.job.timeframe:<3}: {r.metrics['n_trades']:>4} trades, "
              f"{r.metrics['total_return_pct']:>7}% return -> {out_path.name}")

    summary_path = write_summary_report(run_results)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
