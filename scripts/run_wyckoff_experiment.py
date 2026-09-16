"""First real test of the Wyckoff accumulation/VSA idea (see
src/analysis/wyckoff.py, config/strategies/wyckoff_accumulation.yaml) —
same known/virgin two-window discipline, R-multiple stats, and
significance test as every other candidate this project has ever run
(see scripts/run_baseline_experiment.py / run_cost_stress_and_significance.py).

Usage:
    .venv/bin/python scripts/run_wyckoff_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import run_backtest
from src.backtest.experiment import build_trade_log, experiment_report, max_drawdown_r_by_market
from src.backtest.significance import significance_report
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
WINDOWS = [("known", "2023-09-13", "2026-09-15"), ("virgin", "2020-09-01", "2023-09-13")]
STRATEGY_PATH = "config/strategies/wyckoff_accumulation.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    raw_logs, r_logs = [], []

    for label, since, until in WINDOWS:
        print(f"\n=== {label}: {since} -> {until} ===\n")
        print(f"{'market':<10}{'n_trades':>10}{'win_rate%':>11}{'profit_factor':>15}{'expectancy_R':>14}")
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            trades, _ = run_backtest(df, strategy)
            if trades.empty:
                print(f"{market.symbol:<10}{0:>10}")
                continue
            trades = trades.copy()
            trades["market"] = market.symbol
            trades["window"] = label
            raw_logs.append(trades)
            log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
            log["window"] = label
            r_logs.append(log)
            report = experiment_report(log)
            pf = report["profit_factor"]
            pf_str = f"{pf:.2f}" if pf is not None else "n/a"
            print(f"{market.symbol:<10}{report['n_trades']:>10}{report['win_rate_pct']:>11.2f}{pf_str:>15}{report['expectancy_r']:>14.3f}")

    full_r_log = pd.concat(r_logs, ignore_index=True) if r_logs else pd.DataFrame()
    full_raw = pd.concat(raw_logs, ignore_index=True) if raw_logs else pd.DataFrame()

    print("\n=== POOLED (both windows, all 5 markets) ===\n")
    report = experiment_report(full_r_log)
    pf = report["profit_factor"]
    pf_str = f"{pf:.2f}" if pf is not None else "n/a"
    print(f"n={report['n_trades']}  win_rate={report['win_rate_pct']:.2f}%  PF={pf_str}  "
          f"expectancy={report['expectancy_r']:.3f}R  avg_winner={report['avg_winner_r']:.3f}R  avg_loser={report['avg_loser_r']:.3f}R")

    if not full_r_log.empty and "exit_reason" in full_r_log.columns:
        breakdown = full_r_log["exit_reason"].value_counts(normalize=True) * 100
        print(f"exit_reason mix: {breakdown.round(1).to_dict()}")

    if not full_raw.empty:
        sig = significance_report(full_raw)
        print(f"\nsignificance: n={sig['n_trades']}  observed_win_rate={sig['observed_win_rate']}%  "
              f"breakeven={sig['breakeven_win_rate']}%  95% CI=[{sig['ci_95_low']}%, {sig['ci_95_high']}%]  "
              f"breakeven_inside_ci={sig['breakeven_inside_ci']}  p_value={sig['p_value_vs_breakeven']}")

    if not full_r_log.empty:
        print("\nmax drawdown by market (never pooled across markets):")
        print(max_drawdown_r_by_market(full_r_log).to_string(index=False))


if __name__ == "__main__":
    main()
