"""Minimal end-to-end backtest (docs/PLAN.md "Immediate next steps", step 4):
BTC/USDT, 1h timeframe, to validate the full pipeline (data -> signal ->
simulation -> report) before scaling up to 5 markets x 3 timeframes.

Usage:
    .venv/bin/python scripts/run_minimal_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestConfig, compute_metrics, run_backtest
from src.data.fetcher import earliest_available, fetch_ohlcv
from src.report.report import render_report

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE = "2024-06-01"
UNTIL = "2024-12-01"


def main() -> None:
    print(f"Checking listing date for {SYMBOL}...")
    listed_since = earliest_available(SYMBOL, timeframe="1d")
    print(f"  -> first candle available: {listed_since}")

    print(f"Downloading {SYMBOL} {TIMEFRAME} candles from {SINCE} to {UNTIL}...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"  -> {len(df)} candles downloaded")

    if df.empty:
        print("No candles were downloaded — cannot continue.")
        return

    cfg = BacktestConfig()
    trades, equity_curve = run_backtest(df, cfg)
    metrics = compute_metrics(trades, equity_curve, cfg.initial_equity)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out_path = Path(__file__).resolve().parents[1] / "reports" / "minimal_backtest_report.md"
    render_report(SYMBOL, TIMEFRAME, trades, metrics, out_path)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
