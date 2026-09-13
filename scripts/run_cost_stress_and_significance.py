"""Two checks agreed after the volume+candle comparison, on the two
leading candidates (trend_pullback_htf_plus_volume_candle and the full
trend_pullback_htf chain), pooling trades across all 5 markets AND both
windows (known + virgin) for maximum statistical power:

1. Cost stress test: does the result survive doubling the assumed fee
   (a standard industry sanity check, pending since it was first flagged).
2. Formal significance: is the pooled win rate distinguishable from this
   strategy's own breakeven rate, given how many trades we actually have?

See docs/PLAN.md / Bitácora Illari.

Usage:
    .venv/bin/python scripts/run_cost_stress_and_significance.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import run_backtest
from src.backtest.significance import significance_report
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("2023-09-13", "2026-09-13"), ("2020-09-01", "2023-09-13")]

STRATEGY_PATHS = [
    "config/strategies/trend_pullback_htf_plus_volume_candle.yaml",
    "config/strategies/trend_pullback_htf.yaml",
]


def _pooled_trades(strategy, fee_pct: float) -> pd.DataFrame:
    strategy = replace(strategy, fee_pct=fee_pct)
    all_trades = []
    for since, until in WINDOWS:
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            if not trades.empty:
                all_trades.append(trades)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def _pooled_stats(trades: pd.DataFrame) -> dict:
    """Order-independent aggregate stats, safe to compute over trades pooled
    from 10 separate (market, window) backtests. Deliberately does NOT
    include a pooled "total return": that would require treating trades from
    different markets/windows as one sequentially-compounding equity curve,
    which they are not - there's no single account that took all of them in
    this order. Win rate, profit factor and average pnl per trade are all
    plain aggregates over the trade set and carry no such implication."""
    if trades.empty:
        return {"n_trades": 0, "win_rate_pct": 0.0, "profit_factor": None, "avg_pnl_pct": 0.0}

    n = len(trades)
    wins = trades[trades["pnl_abs"] > 0]
    losses = trades[trades["pnl_abs"] <= 0]
    gross_win = wins["pnl_abs"].sum()
    gross_loss = -losses["pnl_abs"].sum()
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    return {
        "n_trades": n,
        "win_rate_pct": round(100 * len(wins) / n, 2),
        "profit_factor": profit_factor,
        "avg_pnl_pct": round(trades["pnl_pct"].mean(), 3),
    }


def main() -> None:
    for path in STRATEGY_PATHS:
        strategy = load_strategy(path)
        print(f"\n{'=' * 70}\n{strategy.name}  (normal fee_pct={strategy.fee_pct})\n{'=' * 70}")

        for label, fee_pct in [("normal cost", strategy.fee_pct), ("DOUBLED cost", strategy.fee_pct * 2)]:
            trades = _pooled_trades(strategy, fee_pct)
            stats = _pooled_stats(trades)
            sig = significance_report(trades)

            print(f"\n  --- {label} (fee_pct={fee_pct}) ---")
            print(f"  pooled trades: {stats['n_trades']}, win rate {stats['win_rate_pct']}%, "
                  f"profit factor {stats['profit_factor']}, avg pnl/trade {stats['avg_pnl_pct']}%")
            print(f"  breakeven win rate: {sig['breakeven_win_rate']}%  |  observed: {sig['observed_win_rate']}%  "
                  f"|  95% CI: [{sig['ci_95_low']}%, {sig['ci_95_high']}%]")
            print(f"  breakeven inside 95% CI? {sig['breakeven_inside_ci']}  |  p-value (one-sided): {sig['p_value_vs_breakeven']}")


if __name__ == "__main__":
    main()
