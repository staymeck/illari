"""Diagnostic: for the current best config (trend_pullback_htf_structural_
stop), checks how often the entry decision could have been made earlier
than the 1h close — or would have fired at all — by watching the bar form
every 5 minutes instead of only at its close.

No P&L, no exits — this only answers "how much is the fixed-hour close
discipline costing us in visibility/timing", before deciding whether a full
intrabar-aware backtest is worth building. See docs/PLAN.md / Bitácora
Illari.

Scoped to the known window (2023-09-13 to 2026-09-13) only: that's the
window 5-minute data is already cached for from the earlier 5m-timeframe
tests; the virgin window would need a fresh multi-year 5m download, left
for later if this looks worth pursuing further.

Usage:
    .venv/bin/python scripts/run_intrabar_entry_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.markets import MARKETS
from src.backtest.intrabar_entry import analyze_intrabar_entries
from src.data.fetcher import fetch_ohlcv
from src.strategies.builder import load_strategy

PARENT_TIMEFRAME = "1h"
CHILD_TIMEFRAME = "5m"
HIGHER_TIMEFRAME = "1d"
SINCE, UNTIL = "2023-09-13", "2026-09-13"

STRATEGY_PATH = "config/strategies/trend_pullback_htf_structural_stop.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    for market in MARKETS:
        parent_df = fetch_ohlcv(market.symbol, PARENT_TIMEFRAME, since=SINCE, until=UNTIL)
        child_df = fetch_ohlcv(market.symbol, CHILD_TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)

        result = analyze_intrabar_entries(parent_df, child_df, strategy, higher_tf_df=higher_tf_df)

        n_hours_checked = len(result)
        n_full_data = (result["n_children"] == 12).sum()
        standard_entries = result[result["standard_signal"]]
        invisible_entries = result[result["intrabar_signal"] & ~result["standard_signal"]]

        print(f"\n{'=' * 70}\n{market.symbol}\n{'=' * 70}")
        print(f"hours checked: {n_hours_checked} (full 12/12 five-min coverage: {n_full_data})")
        print(f"standard (close-based) entries: {len(standard_entries)}")

        if not standard_entries.empty:
            # hour_timestamp is the bar's OPEN, not its close - "minutes
            # earlier" has to be measured against when the bar actually
            # closes (open + the parent timeframe's own duration), not
            # against its open (every intrabar fire is trivially after the
            # open, so that comparison was always negative and meaningless).
            parent_interval = parent_df["timestamp"].iloc[1] - parent_df["timestamp"].iloc[0]
            minutes_earlier = (
                (standard_entries["hour_timestamp"] + parent_interval) - standard_entries["intrabar_fire_time"]
            ).dt.total_seconds() / 60
            price_diff_pct = (
                (standard_entries["intrabar_fire_price"] - standard_entries["close_price"])
                / standard_entries["close_price"]
                * 100
            )
            print(f"  of these, intrabar signal also found: {standard_entries['intrabar_signal'].sum()} "
                  f"(should equal the total above if every hour had full 5m coverage)")
            print(f"  avg minutes earlier the entry could have fired (vs. waiting for the close): "
                  f"{minutes_earlier.mean():.1f} (median {minutes_earlier.median():.1f})")
            print(f"  avg entry price difference vs. waiting for the close: {price_diff_pct.mean():.3f}% "
                  f"(negative = a better, lower price)")

        print(f"signals invisible to the close-based system (fired intrabar, vanished by close): "
              f"{len(invisible_entries)}")


if __name__ == "__main__":
    main()
