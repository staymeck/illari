"""The "tramo virgen" check: 2020-09-01 -> 2023-09-13, a period this
session has NEVER examined (every prior run used 2023-09-13 onward) —
available for all 5 markets (SOL/PAXG listed ~2020-08, the latest of the
5). Run ONCE, no tuning based on what comes back — the whole point is
escaping the "researcher degrees of freedom" problem named in docs/PLAN.md
(iterating many strategy variants against the same historical window is
itself a form of overfitting, even when each individual test is done
properly).

Compares the baseline, the multi-timeframe confirmation (this session's
most consistent finding), and the new regime-adaptive-stop variant, on 1h
across all 5 markets. See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/run_holdout_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.backtest.engine import compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
SINCE = "2020-09-01"  # never examined this session
UNTIL = "2023-09-13"  # exactly where every other run in this session starts

STRATEGY_PATHS = [
    "config/strategies/trend_pullback_fib.yaml",
    "config/strategies/trend_pullback_htf.yaml",
    "config/strategies/trend_pullback_htf_adx.yaml",
]


def main() -> None:
    print(f"=== HOLDOUT VALIDATION: {SINCE} -> {UNTIL} (never examined before this run) ===\n")

    strategies = [load_strategy(p) for p in STRATEGY_PATHS]
    header = "".join(f"{s.name:>28}" for s in strategies)
    print(f"{'market':<10}{header}")

    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)

        row = ""
        for strategy in strategies:
            trades, equity = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            metrics = compute_metrics(trades, equity, strategy.initial_equity)
            row += f"{metrics['n_trades']:>4} / {metrics['total_return_pct']:>8}%   "
        print(f"{market.symbol:<10}{row}")


if __name__ == "__main__":
    main()
