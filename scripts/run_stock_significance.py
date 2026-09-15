"""Formal significance check on the stock-market baseline result (see
scripts/run_stock_baseline_backtest.py) — same question and same method
as scripts/run_cost_stress_and_significance.py already applied to crypto:
is the pooled win rate distinguishable from this strategy's own breakeven
rate, or still plausibly noise, given how many trades we actually have?
Pools RAW trades (not the R-multiple trade log) across all 20 tickers and
both windows for maximum statistical power, same convention.

Usage:
    .venv/bin/python scripts/run_stock_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.stock_markets import STOCK_MARKETS
from src.backtest.engine import run_backtest
from src.backtest.significance import significance_report
from src.data.stock_fetcher import fetch_stock_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("2023-09-13", "2026-09-15"), ("2020-07-27", "2023-09-13")]
STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    all_trades = []
    for since, until in WINDOWS:
        for market in STOCK_MARKETS:
            df = fetch_stock_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_stock_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            if df.empty:
                continue
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            if not trades.empty:
                all_trades.append(trades)

    pooled = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    report = significance_report(pooled)

    print(f"POOLED across {len(STOCK_MARKETS)} stocks x both windows ({strategy.name})\n")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
