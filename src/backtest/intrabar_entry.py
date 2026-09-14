"""Checking a 1h strategy's entries every 5 minutes as it forms, instead of
only at the bar's final close — motivated directly by the discussion
earlier this session: fixed-timeframe analysis always decides "as of the
close", so a real signal that appeared 20 minutes into the hour and
reversed by the close is invisible to it, and even a signal that DOES
survive to the close was, in principle, knowable (and tradeable, at a
better price) earlier.

Two functions, two different questions:

- analyze_intrabar_entries: a DIAGNOSTIC, no P&L — for every 1h bar, was
  the entry condition true at any point while it was forming, and how does
  that compare to the standard close-based decision? Answers "how much is
  the close discipline costing us in visibility/timing".
- run_intrabar_backtest: a REAL backtest — same entry logic, but actually
  executes and manages the resulting trades (stop/target/timeout, fees,
  equity) at 5-minute resolution, reusing the engine's own exit mechanics
  exactly. Answers "does entering earlier actually help, once real exits
  and costs are in the picture" — built only after the diagnostic above
  showed the timing gap was large enough to be worth testing for real (see
  docs/PLAN.md / Bitácora Illari).

Both reuse the engine's own signal-finding logic (_find_entry_signal)
exactly as-is — the only thing that changes is what stands in for "the
current bar": the fully closed bar (matching run_backtest), or a partial
reconstruction as of a given 5-minute mark within it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import (
    _build_trade_record,
    _confirmed_pivots_as_of,
    _find_entry_signal,
    _higher_tf_window_as_of,
    _walk_to_exit,
)
from src.analysis.structure import find_swing_points
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext


def _child_parent_index(parent_df: pd.DataFrame, child_df: pd.DataFrame) -> pd.DataFrame:
    """child_df, filtered to rows that actually fall within some parent_df
    bar's time span and sorted chronologically, with a `_parent_idx` column
    added — the single source of truth both _map_children_to_parents and
    run_intrabar_backtest build on."""
    child_df = child_df.sort_values("timestamp").reset_index(drop=True)
    parent_times = parent_df["timestamp"].to_numpy()
    parent_interval = parent_df["timestamp"].iloc[1] - parent_df["timestamp"].iloc[0]

    bucket = np.searchsorted(parent_times, child_df["timestamp"].to_numpy(), side="right") - 1
    child_df = child_df.assign(_parent_idx=bucket)
    child_df = child_df[child_df["_parent_idx"] >= 0]
    span_end = pd.Series(parent_times[child_df["_parent_idx"]]) + parent_interval
    child_df = child_df[child_df["timestamp"].to_numpy() < span_end.to_numpy()]
    return child_df.reset_index(drop=True)


def _map_children_to_parents(parent_df: pd.DataFrame, child_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Groups child_df rows by which parent_df bar's time span they fall
    in — same technique as src/analysis/intracandle.py."""
    indexed = _child_parent_index(parent_df, child_df)
    return {idx: group for idx, group in indexed.groupby("_parent_idx")}


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


def run_intrabar_backtest(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    strategy: ResolvedStrategy,
    higher_tf_df: pd.DataFrame | None = None,
    higher_tf_lookback_bars: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """A real backtest, not just a diagnostic: entries are decided by
    watching each 1h bar form every 5 minutes (same partial-bar
    reconstruction as analyze_intrabar_entries), executed at the NEXT
    5-minute bar's open — but once entered, the position is walked forward
    using the child (5-minute) bars themselves, reusing the engine's own
    exit/fee/equity mechanics (_walk_to_exit, _build_trade_record) exactly
    as run_backtest does, just at finer resolution. One trade at a time,
    same rule as run_backtest.

    `strategy.max_holding_bars` is defined in units of the PARENT timeframe
    (1h) — converted here to the equivalent number of child (5m) bars.

    Returns `(trades_df, equity_curve)`, same shape as run_backtest's.
    """
    parent_df = parent_df.reset_index(drop=True)
    parent_df = parent_df.astype({"open": float, "high": float, "low": float, "close": float})
    marked_full = find_swing_points(parent_df, order=strategy.swing_order)
    parent_interval = parent_df["timestamp"].iloc[1] - parent_df["timestamp"].iloc[0]

    interval = None
    if higher_tf_df is not None:
        higher_tf_df = higher_tf_df.reset_index(drop=True)
        interval = higher_tf_df["timestamp"].diff().median()

    children = _child_parent_index(parent_df, child_df)
    children = children.astype({"open": float, "high": float, "low": float, "close": float})
    # Precomputed cumulative extreme within each forming parent bar, so a
    # position that exits mid-bar and resumes scanning later in that same
    # bar still sees the true high/low reached before the exit, not a
    # reset-to-open value.
    children["_seen_high"] = children.groupby("_parent_idx")["high"].cummax()
    children["_seen_low"] = children.groupby("_parent_idx")["low"].cummin()

    child_interval = children["timestamp"].diff().median()
    children_per_parent = max(1, round(parent_interval / child_interval))
    max_holding_children = strategy.max_holding_bars * children_per_parent

    high_col = parent_df.columns.get_loc("high")
    low_col = parent_df.columns.get_loc("low")
    close_col = parent_df.columns.get_loc("close")

    n_children = len(children)
    trades: list[dict] = []
    equity = strategy.initial_equity
    equity_curve = [equity]

    j = 0
    while j < n_children - 1:  # need a next child bar to execute on
        row = children.iloc[j]
        p = int(row["_parent_idx"])

        if p < strategy.lookback_bars or p >= len(parent_df) - 1:
            j += 1
            continue

        window_start = max(0, p + 1 - strategy.lookback_bars)
        marked_window = _confirmed_pivots_as_of(marked_full, p, window_start, strategy.swing_order)
        price_window = parent_df.iloc[window_start : p + 1].copy()
        price_window.iloc[-1, high_col] = row["_seen_high"]
        price_window.iloc[-1, low_col] = row["_seen_low"]
        price_window.iloc[-1, close_col] = row["close"]

        htf_window = None
        if higher_tf_df is not None:
            htf_window = _higher_tf_window_as_of(higher_tf_df, row["timestamp"], higher_tf_lookback_bars, interval)
        ctx = EvalContext(price_window=price_window, marked_window=marked_window, higher_tf_window=htf_window)

        signal = _find_entry_signal(ctx, strategy)
        if signal is None:
            j += 1
            continue

        setup, extras = signal
        entry_idx = j + 1  # execute at the NEXT 5-minute bar's open
        entry_price = float(children["open"].iloc[entry_idx])

        stop_price = strategy.stop_fn(entry_price, setup, ctx, strategy.stop_params)
        target_price = strategy.target_fn(entry_price, stop_price, ctx, strategy.target_params)

        exit_idx, exit_price, exit_reason, stop_was_premature = _walk_to_exit(
            children, entry_idx, stop_price, target_price, strategy.stop_trigger, max_holding_children
        )
        trade, equity = _build_trade_record(
            children, entry_idx, exit_idx, entry_price, exit_price, exit_reason, stop_was_premature,
            stop_price, target_price, strategy.fee_pct, equity, extras,
        )
        trades.append(trade)
        equity_curve.append(equity)

        j = exit_idx + 1

    return pd.DataFrame(trades), pd.Series(equity_curve)
