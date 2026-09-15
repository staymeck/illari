"""Runs the SAME frozen baseline strategy (trend_pullback_htf_confluence,
zero retuning) against the 5 stock markets in config/stock_markets.py —
does the crypto-tuned strategy generalize to a structurally different
asset class at all, or was it fit to crypto's own quirks? Same known/
virgin two-window discipline and R-multiple stats as
scripts/run_baseline_experiment.py, for direct comparability.

Data availability constraint (Alpaca free/IEX tier, confirmed empirically,
not assumed): history starts 2020-07-27 for all 5 tickers — so "virgin"
here is 2020-07-27 -> 2023-09-13 (~3.1 years, close to but not identical
to crypto's 2020-09-01 -> 2023-09-13 virgin window). "known" is the exact
same 2023-09-13 -> now as crypto's, for a fair side-by-side.

Usage:
    .venv/bin/python scripts/run_stock_baseline_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.stock_markets import STOCK_MARKETS
from src.backtest.engine import run_backtest
from src.backtest.experiment import build_trade_log, experiment_report
from src.data.stock_fetcher import fetch_stock_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-15"), ("virgin", "2020-07-27", "2023-09-13")]
STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    for label, since, until in WINDOWS:
        print(f"\n=== {label}: {since} -> {until} ===\n")
        print(f"{'market':<10}{'n_trades':>10}{'win_rate%':>11}{'profit_factor':>15}{'expectancy_R':>14}{'avg_winner_R':>14}{'avg_loser_R':>13}")
        logs = []
        for market in STOCK_MARKETS:
            df = fetch_stock_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_stock_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            if df.empty:
                print(f"{market.symbol:<10}  (no data)")
                continue
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
            if not log.empty:
                logs.append(log)
            report = experiment_report(log)
            pf = report["profit_factor"]
            pf_str = f"{pf:.2f}" if pf is not None else "n/a"
            print(
                f"{market.symbol:<10}{report['n_trades']:>10}{report['win_rate_pct']:>11.2f}{pf_str:>15}"
                f"{report['expectancy_r']:>14.3f}{report['avg_winner_r']:>14.3f}{report['avg_loser_r']:>13.3f}"
            )

        pooled = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        report = experiment_report(pooled)
        pf = report["profit_factor"]
        pf_str = f"{pf:.2f}" if pf is not None else "n/a"
        print(f"{'POOLED':<10}{report['n_trades']:>10}{report['win_rate_pct']:>11.2f}{pf_str:>15}{report['expectancy_r']:>14.3f}")


if __name__ == "__main__":
    main()
