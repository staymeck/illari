"""Stage 2 of the experiment protocol: baseline vs. +5M rejection (2A) and
baseline vs. +5M structure-break (2B), each isolated — same setup, same
stop/target, same execution timeframe (1h); the only difference in each
run is exactly one added lower-tf confirmation, evaluated once at the
existing 1h-close decision point (NOT the already-rejected continuous
5-minute-check idea).

Scope note, stated plainly rather than glossed over: known window only
(2023-09-13 to 2026-09-13) — 5-minute data is only cached there from
earlier session work; the virgin window would need a fresh multi-year 5m
download per market. Baseline is recomputed restricted to the same known
window for a fair comparison (Stage 0's original baseline pooled both
windows).

Usage:
    .venv/bin/python scripts/run_stage2_5m_experiment.py
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
LOWER_TIMEFRAME = "5m"
SINCE, UNTIL = "2023-09-13", "2026-09-13"

BASELINE_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"
STAGE2A_PATH = "config/strategies/trend_pullback_htf_confluence_5m_rejection.yaml"
STAGE2B_PATH = "config/strategies/trend_pullback_htf_confluence_5m_structure.yaml"


def _run(strategy_path: str, use_5m: bool) -> pd.DataFrame:
    strategy = load_strategy(strategy_path)
    logs = []
    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)
        lower_tf_df = fetch_ohlcv(market.symbol, LOWER_TIMEFRAME, since=SINCE, until=UNTIL) if use_5m else None
        trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df, lower_tf_df=lower_tf_df)
        log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
        if not log.empty:
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
    baseline_log = _run(BASELINE_PATH, use_5m=False)
    stage2a_log = _run(STAGE2A_PATH, use_5m=True)
    stage2b_log = _run(STAGE2B_PATH, use_5m=True)

    _print_report("BASELINE (known window only)", baseline_log)
    _print_report("2A: +5M rejection", stage2a_log)
    _print_report("2B: +5M structure break", stage2b_log)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stage2a_log.to_csv(REPORTS_DIR / "stage2a_5m_rejection_trade_log.csv", index=False)
    stage2b_log.to_csv(REPORTS_DIR / "stage2b_5m_structure_trade_log.csv", index=False)
    print(f"\nTrade logs saved under {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
