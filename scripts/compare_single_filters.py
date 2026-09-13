""""One filter at a time" comparison: instead of chaining every
confirmation together (which starved trend_pullback_htf_adx down to 0-13
trades over ~6 years — see Bitácora Illari), start from
trend_pullback_htf_minimal.yaml (just the multi-timeframe confirmation) and
add exactly ONE more confirmation per variant, to see which single piece
carries real weight versus which one is just shrinking the sample for
nothing. Runs both the known window (2023-2026) and the never-examined
holdout (2020-2023) for every variant, across all 5 markets, on 1h.

Usage:
    .venv/bin/python scripts/compare_single_filters.py
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

WINDOWS = [
    ("known (2023-26)", "2023-09-13", "2026-09-13"),
    ("virgin (2020-23)", "2020-09-01", "2023-09-13"),
]

STRATEGY_PATHS = [
    "config/strategies/trend_pullback_htf_minimal.yaml",
    "config/strategies/trend_pullback_htf_plus_volume.yaml",
    "config/strategies/trend_pullback_htf_plus_candle.yaml",
    "config/strategies/trend_pullback_htf_plus_volume_candle.yaml",
    "config/strategies/trend_pullback_htf.yaml",  # full chain, for reference
]


def main() -> None:
    strategies = [load_strategy(p) for p in STRATEGY_PATHS]

    for label, since, until in WINDOWS:
        print(f"\n=== {label}: {since} -> {until} ===\n")
        header = "".join(f"{s.name.replace('trend_pullback_', ''):>22}" for s in strategies)
        print(f"{'market':<10}{header}")

        totals = [0] * len(strategies)
        sums = [0.0] * len(strategies)

        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)

            row = ""
            for k, strategy in enumerate(strategies):
                trades, equity = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
                metrics = compute_metrics(trades, equity, strategy.initial_equity)
                row += f"{metrics['n_trades']:>4} / {metrics['total_return_pct']:>7}%"
                totals[k] += metrics["n_trades"]
                sums[k] += metrics["total_return_pct"]
            print(f"{market.symbol:<10}{row}")

        avg_row = "".join(f"{totals[k]:>4} / {sums[k] / len(MARKETS):>7.2f}%" for k in range(len(strategies)))
        print(f"{'TOTAL/AVG':<10}{avg_row}")


if __name__ == "__main__":
    main()
