"""Stage 1 of the experiment protocol: baseline vs. baseline+4H, isolated —
same setup, same stop/target, same execution timeframe (1h), the ONLY
difference is the extra higher_tf_trend_2 (4h) confirmation. Same metrics
as Stage 0 (src/backtest/experiment.py), so the comparison is apples to
apples.

The question this answers (per the user's protocol): does the 4H filter
remove bad trades without destroying too many good ones — profit factor
and expectancy up, without gutting the trade count — or does it barely
change anything while cutting opportunity?

Usage:
    .venv/bin/python scripts/run_stage1_4h_experiment.py
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
HIGHER_TIMEFRAME_2 = "4h"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]

BASELINE_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"
STAGE1_PATH = "config/strategies/trend_pullback_htf_confluence_4h.yaml"


def _run(strategy_path: str, use_4h: bool) -> pd.DataFrame:
    strategy = load_strategy(strategy_path)
    logs = []
    for label, since, until in WINDOWS:
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            higher_tf_df_2 = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME_2, since=since, until=until) if use_4h else None
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df, higher_tf_df_2=higher_tf_df_2)
            log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
            if not log.empty:
                log["window"] = label
                logs.append(log)
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def _print_report(label: str, log: pd.DataFrame) -> None:
    report = experiment_report(log)
    dd = max_drawdown_r_by_market(log)
    print(f"\n--- {label} ---")
    print(f"n_trades: {report['n_trades']}")
    print(f"win_rate: {report['win_rate_pct']}%   loss_rate: {report['loss_rate_pct']}%")
    print(f"profit_factor: {report['profit_factor']}")
    print(f"expectancy: {report['expectancy_r']}R")
    print(f"avg_winner: {report['avg_winner_r']}R   avg_loser: {report['avg_loser_r']}R")
    print("max drawdown by market (R):")
    print(dd.to_string(index=False) if not dd.empty else "  (no trades)")


def main() -> None:
    baseline_log = _run(BASELINE_PATH, use_4h=False)
    stage1_log = _run(STAGE1_PATH, use_4h=True)

    _print_report("BASELINE (no 4H filter)", baseline_log)
    _print_report("+4H filter", stage1_log)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "stage1_4h_trade_log.csv"
    stage1_log.to_csv(out_path, index=False)
    print(f"\nStage 1 trade log saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
