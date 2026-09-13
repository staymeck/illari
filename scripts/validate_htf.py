"""Full validation of the multi-timeframe confirmation (trend_pullback_htf.yaml
vs trend_pullback_fib.yaml): cross-market comparison on 1h across all 5
markets, then walk-forward (3 folds) on whichever markets look promising —
same discipline already applied to the ADX/ATR combos. See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/validate_htf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.backtest.engine import compute_metrics, run_backtest
from src.backtest.walk_forward import chronological_folds
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"
N_FOLDS = 3


def _run_pair(df, higher_tf_df, baseline_strategy, htf_strategy):
    baseline_trades, baseline_equity = run_backtest(df, baseline_strategy)
    baseline_metrics = compute_metrics(baseline_trades, baseline_equity, baseline_strategy.initial_equity)

    htf_trades, htf_equity = run_backtest(df, htf_strategy, higher_tf_df=higher_tf_df)
    htf_metrics = compute_metrics(htf_trades, htf_equity, htf_strategy.initial_equity)
    return baseline_metrics, htf_metrics


def cross_market_check() -> None:
    print(f"=== Cross-market check, {TIMEFRAME} ({SINCE} -> {UNTIL}) ===\n")
    print(f"{'market':<10} {'baseline_n':>10} {'baseline_ret':>12} {'htf_n':>6} {'htf_ret':>9}")

    baseline_strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    htf_strategy = load_strategy("config/strategies/trend_pullback_htf.yaml")

    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)
        baseline_metrics, htf_metrics = _run_pair(df, higher_tf_df, baseline_strategy, htf_strategy)
        print(
            f"{market.symbol:<10} {baseline_metrics['n_trades']:>10} "
            f"{baseline_metrics['total_return_pct']:>11}% {htf_metrics['n_trades']:>6} "
            f"{htf_metrics['total_return_pct']:>8}%"
        )


def walk_forward_check(symbol: str) -> None:
    print(f"\n=== Walk-forward, {symbol} {TIMEFRAME}, {N_FOLDS} folds ===\n")
    baseline_strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    htf_strategy = load_strategy("config/strategies/trend_pullback_htf.yaml")

    baseline_positive = 0
    htf_positive = 0
    for fold in chronological_folds(SINCE, UNTIL, N_FOLDS):
        df = fetch_ohlcv(symbol, TIMEFRAME, since=fold.since, until=fold.until)
        higher_tf_df = fetch_ohlcv(symbol, HIGHER_TIMEFRAME, since=fold.since, until=fold.until)
        baseline_metrics, htf_metrics = _run_pair(df, higher_tf_df, baseline_strategy, htf_strategy)

        if baseline_metrics["total_return_pct"] > 0:
            baseline_positive += 1
        if htf_metrics["total_return_pct"] > 0:
            htf_positive += 1

        print(
            f"  {fold.since} -> {fold.until}: baseline {baseline_metrics['n_trades']:>3} trades / "
            f"{baseline_metrics['total_return_pct']:>7}%   |   htf {htf_metrics['n_trades']:>3} trades / "
            f"{htf_metrics['total_return_pct']:>7}%"
        )

    print(f"\n  baseline: {baseline_positive}/{N_FOLDS} folds positive")
    print(f"  htf:      {htf_positive}/{N_FOLDS} folds positive")


if __name__ == "__main__":
    cross_market_check()
    walk_forward_check("BTC/USDT")
