"""Phase 2, first pass: "what is the market doing right now?" — using only a
small, recent rolling window (not a multi-year historical backtest), the
same way a live system would. Runs trend_pullback_htf.yaml (the session's
most consistent finding) via engine.evaluate_signal on the most recently
closed candle of each of the 5 markets. No execution, no simulated trades —
just a snapshot. See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/market_snapshot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.backtest.engine import evaluate_signal
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
STRATEGY_PATH = "config/strategies/trend_pullback_htf.yaml"

# Reference numbers from this session's own validation (scripts/validate_htf.py,
# 2023-09-13 -> 2026-09-13) — printed for context, not recomputed here, since
# a snapshot's whole point is not needing to re-run the full history.
KNOWN_1H_RESULTS = {
    "BTC/USDT": "7 trades, +3.36% over ~3y",
    "ETH/USDT": "5 trades, -0.24% over ~3y (htf didn't help here)",
    "PAXG/USDT": "26 trades, +1.14% over ~3y",
    "SOL/USDT": "6 trades, -1.47% over ~3y",
    "DOGE/USDT": "5 trades, -0.81% over ~3y",
}


def _recent_window(symbol: str, timeframe: str, bars_needed: int) -> pd.DataFrame:
    """Fetches just enough recent candles to evaluate a signal — a small
    rolling window, not a multi-year history. Adds a buffer so swing
    confirmation (which needs a few extra bars) has room."""
    buffer_bars = bars_needed + 30
    # Rough calendar lookback per bar, generous enough that fetch_ohlcv
    # returns at least buffer_bars candles regardless of timeframe.
    minutes_per_bar = {"5m": 5, "30m": 30, "1h": 60, "1d": 60 * 24}[timeframe]
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes_per_bar * buffer_bars)
    return fetch_ohlcv(symbol, timeframe, since=since.strftime("%Y-%m-%d"))


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    print(f"Snapshot using {strategy.name} ({TIMEFRAME}, confirmed by {HIGHER_TIMEFRAME})\n")

    for market in MARKETS:
        df = _recent_window(market.symbol, TIMEFRAME, strategy.lookback_bars)
        higher_tf_df = _recent_window(market.symbol, HIGHER_TIMEFRAME, 50)

        result = evaluate_signal(df, strategy, higher_tf_df=higher_tf_df)

        status = "SIGNAL — entry conditions met" if result.signal else "no signal"
        print(f"{market.symbol:<10} as of {result.as_of}  trend={result.trend:<10} {status}")
        if result.signal:
            print(f"           reference_level={result.reference_level}  extras={result.extras}")
        print(f"           (for context, this session's 1h backtest: {KNOWN_1H_RESULTS[market.symbol]})")
        print()


if __name__ == "__main__":
    main()
