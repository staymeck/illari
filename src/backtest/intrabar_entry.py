"""Diagnostic (NOT a new backtest mode yet): for every traded timeframe
(1h) bar, checks whether the strategy's entry condition would already have
been true at some point WHILE that bar was still forming — using its
5-minute children to reconstruct the bar's progressive state (open fixed,
high/low/close updating every 5 minutes) — instead of only at the bar's
final close.

Motivated directly by the discussion earlier this session: fixed-timeframe
analysis always decides "as of the close", so a real signal that appeared
20 minutes into the hour and reversed by the close is invisible to it, and
even a signal that DOES survive to the close was, in principle, knowable
earlier. This measures both effects on real data before deciding whether
they're big enough to justify a real intrabar-aware backtest (this module
does NOT simulate exits or P&L — see docs/PLAN.md / Bitácora Illari).

Reuses the engine's own signal-finding logic (_find_entry_signal) exactly
as-is — the only thing that changes is what stands in for "the current
bar": the fully closed bar (matching run_backtest), or a partial
reconstruction as of a given 5-minute mark within it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import _confirmed_pivots_as_of, _find_entry_signal, _higher_tf_window_as_of
from src.analysis.structure import find_swing_points
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext


def _map_children_to_parents(parent_df: pd.DataFrame, child_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Groups child_df rows by which parent_df bar's time span they fall
    in — same technique as src/analysis/intracandle.py."""
    parent_times = parent_df["timestamp"].to_numpy()
    parent_interval = parent_df["timestamp"].iloc[1] - parent_df["timestamp"].iloc[0]

    bucket = np.searchsorted(parent_times, child_df["timestamp"].to_numpy(), side="right") - 1
    child_df = child_df.assign(_parent_idx=bucket)
    child_df = child_df[child_df["_parent_idx"] >= 0]
    span_end = pd.Series(parent_times[child_df["_parent_idx"]]) + parent_interval
    child_df = child_df[child_df["timestamp"].to_numpy() < span_end.to_numpy()]

    return {idx: group.sort_values("timestamp") for idx, group in child_df.groupby("_parent_idx")}


def analyze_intrabar_entries(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    strategy: ResolvedStrategy,
    higher_tf_df: pd.DataFrame | None = None,
    higher_tf_lookback_bars: int = 50,
) -> pd.DataFrame:
    """One row per parent (1h) bar with enough child (5m) coverage to
    check, from `strategy.lookback_bars` onward:

    - standard_signal: does the strategy fire using the bar's real, fully
      closed OHLC (identical to what run_backtest would decide)?
    - intrabar_signal: does it fire at ANY point while the bar was still
      forming?
    - intrabar_fire_time / intrabar_fire_price: the FIRST 5-minute mark
      (if any) where it does, and the price at that moment.
    - n_children: how many 5-minute bars were found for this hour (12
      expected; fewer means a data gap for that hour).

    No P&L, no exits — this only asks "when could the entry decision have
    first been made", not "would it have been profitable"."""
    parent_df = parent_df.reset_index(drop=True)
    # Partial-bar reconstruction writes floats into high/low/close below —
    # cast upfront so an integer-typed OHLCV column (e.g. from synthetic
    # test data) doesn't raise on that assignment.
    parent_df = parent_df.astype({"open": float, "high": float, "low": float, "close": float})
    child_df = child_df.sort_values("timestamp").reset_index(drop=True)
    marked_full = find_swing_points(parent_df, order=strategy.swing_order)

    interval = None
    if higher_tf_df is not None:
        higher_tf_df = higher_tf_df.reset_index(drop=True)
        interval = higher_tf_df["timestamp"].diff().median()

    children_by_parent = _map_children_to_parents(parent_df, child_df)

    high_col = parent_df.columns.get_loc("high")
    low_col = parent_df.columns.get_loc("low")
    close_col = parent_df.columns.get_loc("close")

    records = []
    for i in range(strategy.lookback_bars, len(parent_df) - 1):
        children = children_by_parent.get(i)
        if children is None or children.empty:
            continue

        window_start = max(0, i + 1 - strategy.lookback_bars)
        marked_window = _confirmed_pivots_as_of(marked_full, i, window_start, strategy.swing_order)
        price_window = parent_df.iloc[window_start : i + 1].copy()

        htf_window_full = None
        if higher_tf_df is not None:
            htf_window_full = _higher_tf_window_as_of(
                higher_tf_df, parent_df["timestamp"].iloc[i], higher_tf_lookback_bars, interval
            )
        ctx_full = EvalContext(price_window=price_window, marked_window=marked_window, higher_tf_window=htf_window_full)
        standard_signal = _find_entry_signal(ctx_full, strategy) is not None

        seen_high = float(parent_df["open"].iloc[i])
        seen_low = float(parent_df["open"].iloc[i])
        fire_time = None
        fire_price = None

        for _, child in children.iterrows():
            seen_high = max(seen_high, float(child["high"]))
            seen_low = min(seen_low, float(child["low"]))

            partial_window = price_window.copy()
            partial_window.iloc[-1, high_col] = seen_high
            partial_window.iloc[-1, low_col] = seen_low
            partial_window.iloc[-1, close_col] = float(child["close"])

            htf_window_partial = htf_window_full
            if higher_tf_df is not None:
                htf_window_partial = _higher_tf_window_as_of(
                    higher_tf_df, child["timestamp"], higher_tf_lookback_bars, interval
                )
            ctx_partial = EvalContext(
                price_window=partial_window, marked_window=marked_window, higher_tf_window=htf_window_partial
            )
            if _find_entry_signal(ctx_partial, strategy) is not None:
                fire_time = child["timestamp"]
                fire_price = float(child["close"])
                break

        records.append(
            {
                "bar_index": i,
                "hour_timestamp": parent_df["timestamp"].iloc[i],
                "close_price": float(parent_df["close"].iloc[i]),
                "n_children": len(children),
                "standard_signal": standard_signal,
                "intrabar_signal": fire_time is not None,
                "intrabar_fire_time": fire_time,
                "intrabar_fire_price": fire_price,
            }
        )

    return pd.DataFrame(records)
