"""Walk-forward validation of the two candidates from scripts/compare_strategies.py
that showed a broad or standout improvement over the baseline: does the
result hold up split across 3 consecutive ~12-month folds, or is it driven
by one lucky stretch? See docs/PLAN.md and src/backtest/walk_forward.py.

Usage:
    .venv/bin/python scripts/run_walk_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.walk_forward import run_walk_forward
from src.report.walk_forward import write_walk_forward_report
from src.strategies.builder import load_strategy

SINCE = "2023-09-13"
UNTIL = "2026-09-13"
N_FOLDS = 3

# (symbol, timeframe, strategy_path) — the baseline is included alongside
# each candidate for direct comparison fold-by-fold.
CASES = [
    ("BTC/USDT", "1h", "config/strategies/trend_pullback_fib.yaml"),
    ("BTC/USDT", "1h", "config/strategies/strong_trend_pullback.yaml"),
    ("SOL/USDT", "30m", "config/strategies/trend_pullback_fib.yaml"),
    ("SOL/USDT", "30m", "config/strategies/trend_pullback_atr.yaml"),
    ("SOL/USDT", "30m", "config/strategies/strong_trend_pullback.yaml"),
]


def main() -> None:
    for symbol, timeframe, strategy_path in CASES:
        strategy = load_strategy(strategy_path)
        results = run_walk_forward(symbol, timeframe, strategy, since=SINCE, until=UNTIL, n_folds=N_FOLDS)

        positive = sum(1 for r in results if r.metrics["total_return_pct"] > 0)
        print(f"\n{symbol} {timeframe} {strategy.name}: {positive}/{len(results)} folds positive")
        for r in results:
            m = r.metrics
            print(
                f"  {r.fold.since} -> {r.fold.until}: {m['n_trades']:>3} trades, "
                f"return {m['total_return_pct']:>7}%, PF {m['profit_factor']}"
            )

        out_path = write_walk_forward_report(symbol, timeframe, strategy.name, results)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
