"""Cost-stress test on the stock-market significant result (see
scripts/run_stock_significance.py) — same standard sanity check already
applied to crypto (scripts/run_cost_stress_and_significance.py): does the
result survive doubling the assumed commission? If a result only looks
good at the exact fee assumed, it's fragile, not real.

Usage:
    .venv/bin/python scripts/run_stock_cost_stress.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
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


def _pooled_trades(strategy, fee_pct: float) -> pd.DataFrame:
    strategy = replace(strategy, fee_pct=fee_pct)
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
        # (silences per-fetch "dropped N bar(s)" noise on the 2nd pass by
        # relying on the parquet cache already having the clean data)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    for label, fee_pct in [("baseline fee (0.1%)", strategy.fee_pct), ("doubled fee (0.2%)", strategy.fee_pct * 2)]:
        trades = _pooled_trades(strategy, fee_pct)
        report = significance_report(trades)
        gross_win = trades.loc[trades["pnl_abs"] > 0, "pnl_abs"].sum()
        gross_loss = -trades.loc[trades["pnl_abs"] <= 0, "pnl_abs"].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        print(f"\n=== {label} ===")
        print(f"  n_trades: {report['n_trades']}  win_rate: {report['observed_win_rate']:.2f}%  "
              f"breakeven: {report['breakeven_win_rate']:.2f}%  profit_factor: {pf:.2f}")
        print(f"  95% CI: [{report['ci_95_low']:.2f}%, {report['ci_95_high']:.2f}%]  "
              f"breakeven_inside_ci: {report['breakeven_inside_ci']}  p_value: {report['p_value_vs_breakeven']}")


if __name__ == "__main__":
    main()
