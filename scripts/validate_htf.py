"""Full validation of the multi-timeframe confirmation, plus the combined
htf+session variant: cross-market comparison on 1h across all 5 markets,
then walk-forward (3 folds) on BTC/USDT — same discipline already applied
to the ADX/ATR combos. See docs/PLAN.md.

`higher_tf_df` is passed to every strategy's run_backtest call — strategies
that don't use the higher_tf_trend confirmation simply ignore it, so one
script covers strategies with and without it.

Usage:
    .venv/bin/python scripts/validate_htf.py [timeframe] [higher_timeframe]
    .venv/bin/python scripts/validate_htf.py 30m 1d
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

TIMEFRAME = sys.argv[1] if len(sys.argv) > 1 else "1h"
HIGHER_TIMEFRAME = sys.argv[2] if len(sys.argv) > 2 else "1d"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"
N_FOLDS = 3

STRATEGY_PATHS = [
    "config/strategies/trend_pullback_fib.yaml",
    "config/strategies/trend_pullback_htf.yaml",
    "config/strategies/trend_pullback_htf_adx.yaml",
]


def _run_all(df, higher_tf_df, strategies) -> list[dict]:
    results = []
    for strategy in strategies:
        trades, equity = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
        results.append(compute_metrics(trades, equity, strategy.initial_equity))
    return results


def cross_market_check(strategies) -> None:
    print(f"=== Cross-market check, {TIMEFRAME} ({SINCE} -> {UNTIL}) ===\n")
    header = "".join(f"{s.name:>26}" for s in strategies)
    print(f"{'market':<10}{header}")

    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)
        results = _run_all(df, higher_tf_df, strategies)
        row = "".join(f"{m['n_trades']:>4} / {m['total_return_pct']:>7}%   " for m in results)
        print(f"{market.symbol:<10}{row}")


def walk_forward_check(symbol: str, strategies) -> None:
    print(f"\n=== Walk-forward, {symbol} {TIMEFRAME}, {N_FOLDS} folds ===\n")
    positive_counts = [0] * len(strategies)

    for fold in chronological_folds(SINCE, UNTIL, N_FOLDS):
        df = fetch_ohlcv(symbol, TIMEFRAME, since=fold.since, until=fold.until)
        higher_tf_df = fetch_ohlcv(symbol, HIGHER_TIMEFRAME, since=fold.since, until=fold.until)
        results = _run_all(df, higher_tf_df, strategies)

        for k, m in enumerate(results):
            if m["total_return_pct"] > 0:
                positive_counts[k] += 1

        row = "   |   ".join(f"{s.name}: {m['n_trades']:>3} trades / {m['total_return_pct']:>7}%" for s, m in zip(strategies, results))
        print(f"  {fold.since} -> {fold.until}: {row}")

    print()
    for s, count in zip(strategies, positive_counts):
        print(f"  {s.name}: {count}/{N_FOLDS} folds positive")


if __name__ == "__main__":
    strategies = [load_strategy(p) for p in STRATEGY_PATHS]
    cross_market_check(strategies)
    walk_forward_check("BTC/USDT", strategies)
