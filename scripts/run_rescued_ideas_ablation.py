"""One-filter-at-a-time ablation for the 3 pieces rescued from a
colleague's exploratory branch (probabilities-to-into-trade, David A):
adds each of trendline / marubozu / narrow_range as ONE extra confirmation
on top of the frozen baseline (trend_pullback_htf_confluence), never
stacked together — same discipline as
docs/PLAN.md's staged experiment protocol and scripts/compare_single_filters.py.
R-multiple stats (order-independent, safe to pool — see
src/backtest/experiment.py), known + virgin windows, all 5 markets.

Usage:
    .venv/bin/python scripts/run_rescued_ideas_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import run_backtest
from src.backtest.experiment import build_trade_log, experiment_report
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]

STRATEGY_PATHS = [
    "config/strategies/trend_pullback_htf_confluence.yaml",  # baseline, for reference
    "config/strategies/trend_pullback_htf_confluence_trendline.yaml",
    "config/strategies/trend_pullback_htf_confluence_marubozu.yaml",
    "config/strategies/trend_pullback_htf_confluence_narrow_range.yaml",
]


def main() -> None:
    strategies = [load_strategy(p) for p in STRATEGY_PATHS]

    for label, since, until in WINDOWS:
        print(f"\n=== {label}: {since} -> {until} ===\n")
        print(f"{'variant':<45}{'n_trades':>10}{'win_rate%':>11}{'profit_factor':>15}{'expectancy_R':>14}")
        for strategy in strategies:
            logs = []
            for market in MARKETS:
                df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
                higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
                trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
                log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
                if not log.empty:
                    logs.append(log)
            full_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
            report = experiment_report(full_log)
            pf = report["profit_factor"]
            pf_str = f"{pf:.2f}" if pf is not None else "n/a"
            variant = strategy.name.replace("trend_pullback_htf_confluence", "confluence")
            print(f"{variant:<45}{report['n_trades']:>10}{report['win_rate_pct']:>11.2f}{pf_str:>15}{report['expectancy_r']:>14.3f}")


if __name__ == "__main__":
    main()
