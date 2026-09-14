"""Trade-by-trade audit: for every single trade trend_pullback_htf (the
full chain — context + higher-tf trend + Fibonacci + candlestick + volume)
actually took, shows what called the trend, which pattern fired, and
exactly why it exited when it did — not just the aggregate win rate.

Motivated directly by the request to inspect individual decisions case by
case, especially after the candlestick catalog expansion changed this
strategy's behavior (see docs/PLAN.md / Bitácora Illari).

Writes one markdown report per market x window under reports/, plus a
pooled pattern breakdown across everything, printed to stdout.

Usage:
    .venv/bin/python scripts/run_trade_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import run_backtest
from src.data.fetcher import fetch_ohlcv
from src.report.paths import REPORTS_DIR
from src.report.trade_audit import pattern_breakdown, render_trade_audit_markdown
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]

STRATEGY_PATH = "config/strategies/trend_pullback_htf.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    all_trades = []
    sections = []

    for label, since, until in WINDOWS:
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)

            title = f"{market.symbol} — {label} window ({since} to {until})"
            sections.append(render_trade_audit_markdown(trades, title))
            if not trades.empty:
                trades = trades.copy()
                trades["market"] = market.symbol
                trades["window"] = label
                all_trades.append(trades)

    pooled = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    breakdown = pattern_breakdown(pooled)

    lines = [
        f"# Trade audit — {strategy.name}",
        "",
        f"context_trend and higher_tf_trend always read \"uptrend\" here — that's the",
        f"gate this strategy requires to enter at all, not something that varies per",
        f"trade. pattern, exit_reason and stop_was_premature are what actually differ",
        f"trade to trade, and are the ones worth reading closely below.",
        "",
        "## Pattern breakdown (pooled across all 5 markets, both windows)",
        "",
        breakdown.to_markdown(index=False) if not breakdown.empty else "_no trades_",
        "",
        "## Per-trade detail",
        "",
    ] + sections

    out_path = REPORTS_DIR / "trade_audit_trend_pullback_htf.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Total pooled trades: {len(pooled)}")
    print(breakdown.to_string(index=False) if not breakdown.empty else "no trades")
    print(f"\nFull report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
