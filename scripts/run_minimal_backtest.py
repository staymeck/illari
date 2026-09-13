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
from src.report.render import write_run_report
from src.report.summary import RunResult, write_summary_report

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SCENARIO = "minimal_validation_2024h2"
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

    out_path = write_run_report(SYMBOL, TIMEFRAME, trades, metrics, scenario=SCENARIO)
    print(f"\nReport written to {out_path}")

    summary_path = write_summary_report([RunResult(SYMBOL, TIMEFRAME, metrics, scenario=SCENARIO)])
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
