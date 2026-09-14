"""Visual companion to the opportunity scan: a candlestick chart contrasting
what trend_pullback_htf_minimal actually traded against the oracle's ideal
round trips (src/analysis/opportunity_scan.find_ideal_trades) - built to
answer, visually, "what are we missing, and is some of that raw opportunity
actually just noise?" (see docs/PLAN.md / Bitácora Illari).

Computes the real trades and the ideal trades over the FULL history (for a
correct, well-warmed-up signal and a representative oracle sample), then
displays only a recent, readable window on the chart (the full 6-year, 1h
candle count would be unreadable and would bloat the HTML file).

Usage:
    .venv/bin/python scripts/run_opportunity_chart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.analysis.opportunity_scan import find_ideal_trades
from src.backtest.engine import run_backtest
from src.data.fetcher import fetch_ohlcv
from src.report.chart import write_opportunity_chart
from src.strategies.builder import load_strategy

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
FULL_SINCE, FULL_UNTIL = "2020-09-01", "2026-09-13"

# The candlestick display window: the full 6-year, ~53k-candle history would
# be unreadable and bloat the HTML - a recent, representative slice instead.
DISPLAY_SINCE = "2026-06-15"

# Oracle "ideal trade" parameters - deliberately more selective than the
# statistics run (which used 0.5%-5% thresholds to map the whole curve):
# here we want a HANDFUL of genuinely distinct, plottable swings, not a
# smear covering the whole chart. See find_ideal_trades' docstring.
IDEAL_HORIZON_BARS = 24        # 1 day - matches every strategy's max_holding_bars.
IDEAL_THRESHOLD_PCT = 3.0      # a comfortably-worth-trading move.
IDEAL_MIN_DISTANCE_BARS = 24   # at least a day apart - one marker per distinct swing.

STRATEGY_PATH = "config/strategies/trend_pullback_htf_minimal.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=FULL_SINCE, until=FULL_UNTIL)
    higher_tf_df = fetch_ohlcv(SYMBOL, HIGHER_TIMEFRAME, since=FULL_SINCE, until=FULL_UNTIL)

    real_trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
    ideal_trades = find_ideal_trades(
        df,
        horizon_bars=IDEAL_HORIZON_BARS,
        threshold_pct=IDEAL_THRESHOLD_PCT,
        min_distance_bars=IDEAL_MIN_DISTANCE_BARS,
    )

    display_start = pd.Timestamp(DISPLAY_SINCE, tz="UTC")
    display_df = df[df["timestamp"] >= display_start].reset_index(drop=True)
    display_real = real_trades[real_trades["entry_time"] >= display_start] if not real_trades.empty else real_trades
    display_ideal = ideal_trades[ideal_trades["entry_time"] >= display_start] if not ideal_trades.empty else ideal_trades

    print(f"Full history: {len(df)} bars, {FULL_SINCE} -> {FULL_UNTIL}")
    print(f"Real trades (trend_pullback_htf_minimal) in full history: {len(real_trades)}")
    print(f"Ideal trades (oracle, >= {IDEAL_THRESHOLD_PCT}% in {IDEAL_HORIZON_BARS}h, "
          f">= {IDEAL_MIN_DISTANCE_BARS}h apart) in full history: {len(ideal_trades)}")
    print(f"\nDisplay window: {DISPLAY_SINCE} -> now ({len(display_df)} candles)")
    print(f"  real trades shown: {len(display_real)}")
    print(f"  ideal trades shown: {len(display_ideal)} "
          f"({(display_ideal['direction'] == 'long').sum() if not display_ideal.empty else 0} long, "
          f"{(display_ideal['direction'] == 'short').sum() if not display_ideal.empty else 0} short)")

    out_path = write_opportunity_chart(display_df, display_real, display_ideal, SYMBOL, TIMEFRAME)
    print(f"\nWrote chart to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
