"""Sweeps config/strategies/probability_pullback.yaml's min_probability
threshold from 50% to 75%, to see how sample size and results trade off as
the confidence bar rises — instead of guessing one number. See docs/PLAN.md
and src/analysis/probability_table.py's honesty caveat (in-sample table,
not yet walk-forward validated — this sweep is about finding a threshold
worth validating further, not a conclusion on its own).

Usage:
    .venv/bin/python scripts/sweep_probability_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.probability_table import build_frequency_table
from src.backtest.engine import compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def main() -> None:
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df)} candles, frequency table built (in-sample)\n")

    print(f"{'threshold':>10} {'n_trades':>9} {'win_rate':>9} {'PF':>6} {'return':>9}")
    for threshold in THRESHOLDS:
        strategy = load_strategy("config/strategies/probability_pullback.yaml")
        strategy.context_params["table"] = table
        strategy.context_params["min_probability"] = threshold

        trades, equity = run_backtest(df, strategy)
        metrics = compute_metrics(trades, equity, strategy.initial_equity)
        print(
            f"{threshold:>9.0%} {metrics['n_trades']:>9} {metrics['win_rate_pct']:>8}% "
            f"{metrics['profit_factor']:>6} {metrics['total_return_pct']:>8}%"
        )


if __name__ == "__main__":
    main()
