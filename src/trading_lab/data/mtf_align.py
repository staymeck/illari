"""Aligns candles from an entry timeframe (e.g. 5m) with the last already
closed candle of higher timeframes (e.g. 1h, 1D), without looking ahead.

OHLCV convention: `timestamp` is the OPENING instant of the candle. A
candle with timestamp `t` and duration `d` is only "closed" (with all its
information available) at `t + d`. So, to decide something at the close of
an entry candle (`t_e + d_e`), only a higher-timeframe candle whose close
(`t_h + d_h`) is <= that instant can be used.
"""

from __future__ import annotations

import pandas as pd

_TIMEFRAME_TO_TIMEDELTA = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        return _TIMEFRAME_TO_TIMEDELTA[timeframe.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def with_close_time(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Adds a `close_time` column = timestamp (open) + candle duration."""
    out = df.copy()
    out["close_time"] = out["timestamp"] + timeframe_delta(timeframe)
    return out


def align_higher_timeframe(
    entry_df: pd.DataFrame,
    entry_timeframe: str,
    higher_df: pd.DataFrame,
    higher_timeframe: str,
    prefix: str,
) -> pd.DataFrame:
    """Adds, to each candle of `entry_df`, the columns of the last candle of
    `higher_df` that was already closed when the entry candle closes.

    Uses `merge_asof` with "backward" direction: it can never match a
    higher-timeframe candle whose `close_time` is later than the entry
    candle's `close_time`, so it never looks at a candle still forming.
    """
    entry = with_close_time(entry_df, entry_timeframe).sort_values("close_time")
    higher = with_close_time(higher_df, higher_timeframe).sort_values("close_time")
    higher_renamed = higher.add_prefix(f"{prefix}_")

    merged = pd.merge_asof(
        entry,
        higher_renamed,
        left_on="close_time",
        right_on=f"{prefix}_close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.reset_index(drop=True)


def align_multi_timeframe(
    entry_df: pd.DataFrame,
    entry_timeframe: str,
    structure_df: pd.DataFrame,
    structure_timeframe: str,
    bias_df: pd.DataFrame,
    bias_timeframe: str,
) -> pd.DataFrame:
    """Combines entry (5m) + structure (1h) + bias (1D) into a single
    DataFrame indexed by entry candle, each row with the higher-timeframe
    context that was already closed at that point (no lookahead)."""
    merged = align_higher_timeframe(entry_df, entry_timeframe, structure_df, structure_timeframe, prefix="struct")
    merged = align_higher_timeframe(merged, entry_timeframe, bias_df, bias_timeframe, prefix="bias")
    return merged
