"""Leave-one-ticker-out sensitivity check on the significant stock result
(scripts/run_stock_significance.py: n=141, win_rate=55.32% vs breakeven
42.58%, p=0.0016) — is that result broad-based across the 20-ticker
catalog, or is one or two tickers (like the CAT profit_factor=596/n=2
artifact already flagged in scripts/run_stock_baseline_backtest.py's own
output) doing most of the work?

For each ticker, recomputes the pooled significance test with that ticker
EXCLUDED — a ticker whose removal makes the result notably weaker (or
notably stronger) is either propping up the result or being suppressed by
it; a broad-based result should barely move when any single ticker is
dropped.

Usage:
    .venv/bin/python scripts/run_stock_outlier_check.py
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

    trades_by_ticker: dict[str, pd.DataFrame] = {}
    for market in STOCK_MARKETS:
        per_ticker = []
        for since, until in WINDOWS:
            df = fetch_stock_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_stock_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            if df.empty:
                continue
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            if not trades.empty:
                per_ticker.append(trades)
        trades_by_ticker[market.symbol] = pd.concat(per_ticker, ignore_index=True) if per_ticker else pd.DataFrame()

    full_pool = pd.concat(trades_by_ticker.values(), ignore_index=True)
    full_report = significance_report(full_pool)
    print(f"FULL POOL (all {len(STOCK_MARKETS)} tickers): n={full_report['n_trades']}  "
          f"win_rate={full_report['observed_win_rate']:.2f}%  p={full_report['p_value_vs_breakeven']}\n")

    print(f"{'excluded':<10}{'n_trades':>10}{'win_rate%':>11}{'p_value':>10}{'sig_at_5%':>11}{'delta_win_rate_pp':>20}")
    rows = []
    for symbol, own_trades in trades_by_ticker.items():
        if own_trades.empty:
            continue
        rest = pd.concat([t for s, t in trades_by_ticker.items() if s != symbol], ignore_index=True)
        report = significance_report(rest)
        delta = report["observed_win_rate"] - full_report["observed_win_rate"]
        sig = report["p_value_vs_breakeven"] is not None and report["p_value_vs_breakeven"] < 0.05
        rows.append((symbol, report["n_trades"], report["observed_win_rate"], report["p_value_vs_breakeven"], sig, delta))

    rows.sort(key=lambda r: abs(r[5]), reverse=True)
    for symbol, n, win_rate, p_value, sig, delta in rows:
        p_str = f"{p_value:.4f}" if p_value is not None else "n/a"
        print(f"{symbol:<10}{n:>10}{win_rate:>11.2f}{p_str:>10}{str(sig):>11}{delta:>+20.2f}")

    print(f"\nStill significant at 5% after removing ANY single ticker: {all(r[4] for r in rows)}")
    print(f"Biggest single-ticker swing on pooled win rate: {rows[0][0]} ({rows[0][5]:+.2f}pp)")


if __name__ == "__main__":
    main()
