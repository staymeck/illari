"""Runs multiple independent backtests in parallel worker processes.

The 15 (market, timeframe) combinations of the Phase 1 scale-up (5 markets x
3 timeframes) are fully independent of each other, so this is a case for
plain CPU parallelism — NOT GPU/CUDA. The backtest loop itself is sequential
and stateful (whether we're in a trade at bar `t` depends on bar `t - 1`, and
trades can't overlap), which doesn't vectorize onto a GPU; but running many
independent backtests side by side on separate CPU cores parallelizes for
free, with no change to the algorithm itself.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.backtest.engine import BacktestConfig, compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv

FetchFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class BacktestJob:
    symbol: str
    timeframe: str
    since: str
    until: str
    scenario: str = "default"
    config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass(frozen=True)
class BacktestJobResult:
    job: BacktestJob
    trades: pd.DataFrame
    metrics: dict


def _run_job(job: BacktestJob, fetch_fn: FetchFn) -> BacktestJobResult:
    """Executed inside a worker process: fetch that job's data (cached to disk
    per src/data/fetcher.py) and run its backtest in isolation."""
    df = fetch_fn(job.symbol, job.timeframe, since=job.since, until=job.until)
    trades, equity_curve = run_backtest(df, job.config)
    metrics = compute_metrics(trades, equity_curve, job.config.initial_equity)
    return BacktestJobResult(job=job, trades=trades, metrics=metrics)


def run_jobs_in_parallel(
    jobs: list[BacktestJob],
    fetch_fn: FetchFn = fetch_ohlcv,
    max_workers: int | None = None,
) -> list[BacktestJobResult]:
    """Runs each job in its own worker process and collects the results.

    `fetch_fn` defaults to the real Binance fetcher but can be swapped (e.g.
    in tests) for anything with the same signature — it must be a top-level,
    picklable function, since it's sent to worker processes as-is.
    `max_workers=None` uses all available CPU cores.
    """
    results: list[BacktestJobResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_job, job, fetch_fn) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results
