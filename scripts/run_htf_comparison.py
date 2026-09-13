"""Compares trend_pullback_fib.yaml against trend_pullback_htf.yaml (the
same strategy plus a 1d trend confirmation) on the same market, to see
whether requiring the higher timeframe to agree helps — Kaufman ch. 19.
See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/run_htf_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"


def main() -> None:
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    higher_tf_df = fetch_ohlcv(SYMBOL, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"{SYMBOL}: {len(df)} {TIMEFRAME} candles, {len(higher_tf_df)} {HIGHER_TIMEFRAME} candles\n")

    baseline = load_strategy("config/strategies/trend_pullback_fib.yaml")
    baseline_trades, baseline_equity = run_backtest(df, baseline)
    baseline_metrics = compute_metrics(baseline_trades, baseline_equity, baseline.initial_equity)
    print(f"baseline (no htf):  {baseline_metrics}")

    htf_strategy = load_strategy("config/strategies/trend_pullback_htf.yaml")
    htf_trades, htf_equity = run_backtest(df, htf_strategy, higher_tf_df=higher_tf_df)
    htf_metrics = compute_metrics(htf_trades, htf_equity, htf_strategy.initial_equity)
    print(f"with 1d confirm:    {htf_metrics}")


if __name__ == "__main__":
    main()
