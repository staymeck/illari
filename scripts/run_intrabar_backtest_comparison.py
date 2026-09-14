"""The real test: does entering via 5-minute checks (instead of waiting for
the 1h close) actually improve results once exits and costs are simulated,
or was the earlier timing-visibility diagnostic an illusion?

Compares run_backtest (standard, 1h close-based) against
run_intrabar_backtest (5-minute entry timing) on the current best config,
across all 5 markets. Known window only (5m data cached there already from
the earlier 5m tests) — see docs/PLAN.md / Bitácora Illari.

Usage:
    .venv/bin/python scripts/run_intrabar_backtest_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.backtest.engine import compute_metrics, run_backtest
from src.backtest.intrabar_entry import run_intrabar_backtest
from src.backtest.significance import significance_report
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

PARENT_TIMEFRAME = "1h"
CHILD_TIMEFRAME = "5m"
HIGHER_TIMEFRAME = "1d"
SINCE, UNTIL = "2023-09-13", "2026-09-13"

STRATEGY_PATH = "config/strategies/trend_pullback_htf_structural_stop.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    totals = {"standard": [0, 0.0], "intrabar": [0, 0.0]}
    all_std_trades = []
    all_intra_trades = []

    for market in MARKETS:
        parent_df = fetch_ohlcv(market.symbol, PARENT_TIMEFRAME, since=SINCE, until=UNTIL)
        child_df = fetch_ohlcv(market.symbol, CHILD_TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)

        std_trades, std_equity = run_backtest(parent_df, strategy, higher_tf_df=higher_tf_df)
        std_metrics = compute_metrics(std_trades, std_equity, strategy.initial_equity)

        intra_trades, intra_equity = run_intrabar_backtest(parent_df, child_df, strategy, higher_tf_df=higher_tf_df)
        intra_metrics = compute_metrics(intra_trades, intra_equity, strategy.initial_equity)

        print(f"{market.symbol:<10} standard: {std_metrics['n_trades']:>4} / {std_metrics['total_return_pct']:>7}%   "
              f"intrabar: {intra_metrics['n_trades']:>4} / {intra_metrics['total_return_pct']:>7}%")

        totals["standard"][0] += std_metrics["n_trades"]
        totals["standard"][1] += std_metrics["total_return_pct"]
        totals["intrabar"][0] += intra_metrics["n_trades"]
        totals["intrabar"][1] += intra_metrics["total_return_pct"]
        if not std_trades.empty:
            all_std_trades.append(std_trades)
        if not intra_trades.empty:
            all_intra_trades.append(intra_trades)

    print(f"\nTOTAL/AVG  standard: {totals['standard'][0]:>4} / {totals['standard'][1] / len(MARKETS):>7.2f}%   "
          f"intrabar: {totals['intrabar'][0]:>4} / {totals['intrabar'][1] / len(MARKETS):>7.2f}%")

    import pandas as pd

    for label, trades_list in [("standard", all_std_trades), ("intrabar", all_intra_trades)]:
        pooled = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
        sig = significance_report(pooled)
        print(f"\n{label} pooled significance: n={sig['n_trades']}, observed win rate={sig['observed_win_rate']}%, "
              f"breakeven={sig['breakeven_win_rate']}%, 95% CI=[{sig['ci_95_low']}%, {sig['ci_95_high']}%], "
              f"breakeven inside CI? {sig['breakeven_inside_ci']}")


if __name__ == "__main__":
    main()
