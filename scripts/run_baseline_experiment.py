"""Stage 0 of the experiment protocol: freeze the current best strategy
(trend_pullback_htf_confluence) as the fixed baseline. Saves the full
trade log (CSV, per docs/PLAN.md / Bitácora Illari's field list) and
prints the summary stats every later stage gets compared against.

Usage:
    .venv/bin/python scripts/run_baseline_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import run_backtest
from src.backtest.experiment import build_trade_log, experiment_report, max_drawdown_r_by_market
from src.data.fetcher import fetch_ohlcv
from src.report.paths import REPORTS_DIR
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]

STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    logs = []

    for label, since, until in WINDOWS:
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
            if not log.empty:
                log["window"] = label
                logs.append(log)

    full_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "baseline_trade_log.csv"
    full_log.to_csv(log_path, index=False)

    report = experiment_report(full_log)
    dd = max_drawdown_r_by_market(full_log)

    print(f"BASELINE — {strategy.name}\n")
    print(f"n_trades: {report['n_trades']}")
    print(f"win_rate: {report['win_rate_pct']}%   loss_rate: {report['loss_rate_pct']}%")
    print(f"profit_factor: {report['profit_factor']}")
    print(f"expectancy: {report['expectancy_r']}R")
    print(f"avg_winner: {report['avg_winner_r']}R   avg_loser: {report['avg_loser_r']}R")
    print("\nmax drawdown by market (in R, each market's own chronological sequence):")
    print(dd.to_string(index=False))
    print(f"\nFull trade log saved: {log_path.resolve()}")


if __name__ == "__main__":
    main()
