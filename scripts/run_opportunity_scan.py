"""Opportunity scan across all 5 markets, full history (2020-09-01 to
2026-09-13, both windows combined - this is a descriptive/oracle pass, not
a validated strategy, so there's no holdout to protect here).

Answers: forget our filters for a moment - how often did a real,
cost-clearing profit opportunity exist at all, in either direction, within
a realistic holding horizon? And how does that compare to how often
trend_pullback_htf_minimal (the base multi-timeframe signal, no extra
filters) actually fired?

See docs/PLAN.md / Bitácora Illari - run after the user flagged that live
signals felt too rare to be trustworthy ("1 cada 8 semanas... el filtro no
esta pasando bien").

Usage:
    .venv/bin/python scripts/run_opportunity_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.analysis.opportunity_scan import opportunity_rate
from src.backtest.engine import compute_metrics, run_backtest
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
SINCE, UNTIL = "2020-09-01", "2026-09-13"

# 8h, 1 day (matches every strategy's max_holding_bars=24), 3 days, 1 week.
HORIZONS_BARS = [8, 24, 72, 168]
# 0.5%: barely clears round-trip cost (0.1-0.2% normal, more with slippage).
# 1-3%: comfortably worth trading. 5%: a genuinely big move.
THRESHOLDS_PCT = [0.5, 1.0, 2.0, 3.0, 5.0]

REFERENCE_STRATEGY = "config/strategies/trend_pullback_htf_minimal.yaml"


def main() -> None:
    strategy = load_strategy(REFERENCE_STRATEGY)

    print(f"Opportunity scan: {SINCE} -> {UNTIL}, {TIMEFRAME}, {len(MARKETS)} markets\n")
    print("Reference for comparison: how often trend_pullback_htf_minimal actually")
    print("fired an entry, over the same period - to see what fraction of the raw")
    print("opportunity below it actually captures.\n")

    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)

        trades, equity = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
        metrics = compute_metrics(trades, equity, strategy.initial_equity)

        print(f"{'=' * 78}\n{market.symbol}  ({len(df)} bars)  "
              f"trend_pullback_htf_minimal fired {metrics['n_trades']} entries "
              f"over this period\n{'=' * 78}")

        header = "".join(f"{h}h".rjust(9) for h in HORIZONS_BARS)
        print(f"{'threshold':<12}{header}   (either-direction opportunity rate, % of bars)")

        for threshold in THRESHOLDS_PCT:
            row = ""
            for horizon in HORIZONS_BARS:
                report = opportunity_rate(df, horizon_bars=horizon, threshold_pct=threshold)
                row += f"{report['either_opportunity_pct']:>8.1f}%"
            print(f">= {threshold:>5.1f}%  {row}")

        # Magnitude context at the horizon that matches every strategy's
        # actual max_holding_bars (24h = 1 day on this timeframe).
        mag = opportunity_rate(df, horizon_bars=24, threshold_pct=0.5)
        print(f"\n  at 24h horizon: mean long MFE {mag['mean_mfe_long_pct']}% "
              f"(median {mag['median_mfe_long_pct']}%), "
              f"mean short MFE {mag['mean_mfe_short_pct']}% (median {mag['median_mfe_short_pct']}%)")
        print()


if __name__ == "__main__":
    main()
