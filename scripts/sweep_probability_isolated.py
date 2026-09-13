"""Isolated comparison: does probability_trend carry any signal on its own,
separated from whether it survives being stacked under the full
Fibonacci/candlestick/volume confirmation set? Runs
trend_pullback_isolated.yaml (dow_trend, deterministic baseline) once, and
probability_isolated.yaml across the same 50%-75% threshold sweep as
before — both with ONLY support_touch as the setup, no other
confirmations. See docs/PLAN.md and the session discussion that motivated
this (the earlier sweep on the full stack showed a cliff: garbage trades
below 55%, zero trades above it).

Usage:
    .venv/bin/python scripts/sweep_probability_isolated.py
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


def _print_row(label: str, metrics: dict) -> None:
    print(
        f"{label:>12} {metrics['n_trades']:>9} {metrics['win_rate_pct']:>8}% "
        f"{metrics['profit_factor']:>6} {metrics['total_return_pct']:>8}%"
    )


def main() -> None:
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df)} candles — isolated (support_touch only, no other confirmations)\n")

    print(f"{'version':>12} {'n_trades':>9} {'win_rate':>9} {'PF':>6} {'return':>9}")

    baseline = load_strategy("config/strategies/trend_pullback_isolated.yaml")
    baseline_trades, baseline_equity = run_backtest(df, baseline)
    baseline_metrics = compute_metrics(baseline_trades, baseline_equity, baseline.initial_equity)
    _print_row("dow_trend", baseline_metrics)

    for threshold in THRESHOLDS:
        strategy = load_strategy("config/strategies/probability_isolated.yaml")
        strategy.context_params["table"] = table
        strategy.context_params["min_probability"] = threshold

        trades, equity = run_backtest(df, strategy)
        metrics = compute_metrics(trades, equity, strategy.initial_equity)
        _print_row(f"P>={threshold:.0%}", metrics)


if __name__ == "__main__":
    main()
