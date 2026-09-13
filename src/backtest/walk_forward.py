"""Chronological fold-based out-of-sample validation: instead of judging a
strategy by one aggregate metric over the whole test window, split the
window into N consecutive periods and check whether performance holds up
in each one independently.

This is the practical form "walk-forward" takes here: our strategies use
fixed, hand-picked parameters (never optimized/fit to data), so there is no
per-fold re-optimization step — the same strategy config runs unmodified on
each fold. What this catches is regime-dependency: a strategy whose
aggregate return is driven by one lucky stretch, with the other folds flat
or negative, is not the same as one that holds up consistently — exactly
the distinction Kaufman (Trading Systems and Methods, ch. 21, "Testing
across a Wide Range of Markets" / "Retesting for Changing Parameters")
warns is invisible in a single aggregate number. See docs/PLAN.md.

Caveat: each fold is backtested on its own exact calendar range, so the
first `strategy.lookback_bars` bars of every fold produce no signals (not
yet enough trailing history within the fold) — a small, disclosed warm-up
gap, not a bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtest.engine import compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import ResolvedStrategy

FetchFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class Fold:
    since: str
    until: str


def chronological_folds(since: str, until: str, n_folds: int) -> list[Fold]:
    """Splits `[since, until)` into `n_folds` consecutive, equal-length
    calendar periods."""
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    since_ts = pd.Timestamp(since, tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC")
    edges = pd.date_range(since_ts, until_ts, periods=n_folds + 1)
    return [
        Fold(since=edges[i].strftime("%Y-%m-%d"), until=edges[i + 1].strftime("%Y-%m-%d"))
        for i in range(n_folds)
    ]


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    metrics: dict


def run_walk_forward(
    symbol: str,
    timeframe: str,
    strategy: ResolvedStrategy,
    since: str,
    until: str,
    n_folds: int = 3,
    fetch_fn: FetchFn = fetch_ohlcv,
) -> list[FoldResult]:
    """Runs `strategy`, completely unmodified, independently on each of
    `n_folds` consecutive calendar periods spanning `[since, until)`."""
    results = []
    for fold in chronological_folds(since, until, n_folds):
        df = fetch_fn(symbol, timeframe, since=fold.since, until=fold.until)
        trades, equity_curve = run_backtest(df, strategy)
        metrics = compute_metrics(trades, equity_curve, strategy.initial_equity)
        results.append(FoldResult(fold=fold, metrics=metrics))
    return results
