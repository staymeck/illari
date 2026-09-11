"""Market structure: swing highs/lows, trend, and breaks of structure (BOS).

A swing high on candle `i` is a local maximum: its `high` is greater than
the `high` of the `lookback` candles before and after it. Analogous for a
swing low with `low`. These points are the geometric basis for almost
everything else (Fibonacci anchors to the last swing, trend is defined by
the sequence of swings).

Important: a swing on candle `i` can only be confirmed once the `lookback`
candles after it are known — i.e. it's confirmed with a `lookback`-candle
delay. `find_swings` already reflects that delay: the `swing_high`/
`swing_low` column on row `i` is True only if, looking up to candle `i`,
that swing is already confirmable (it doesn't use data from beyond the
"detection" date, which is reported in `confirmed_at_index`).
"""

from __future__ import annotations

import pandas as pd


def find_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Marks swing highs and swing lows in `df` (high/low columns).

    Returns a copy of `df` with two new boolean columns: `swing_high` and
    `swing_low`, on the candle where the extreme occurs (not the candle
    where it's confirmed). A consumer that wants to avoid lookahead should
    use candle `i + lookback` onward as the moment this information is
    actually available.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    swing_high = [False] * n
    swing_low = [False] * n

    highs = out["high"].to_numpy()
    lows = out["low"].to_numpy()

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]
        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            swing_high[i] = True
        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            swing_low[i] = True

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    # Index of the candle on which this swing becomes confirmable without
    # lookahead (`lookback` candles after it occurred).
    out["confirmed_at_index"] = out.index + lookback
    return out


def trend_from_swings(df_with_swings: pd.DataFrame) -> pd.DataFrame:
    """Derives the prevailing trend on each candle from the sequence of
    already-confirmed swings (no lookahead): 'up' if the latest pair of
    swings forms higher highs and higher lows, 'down' if lower, 'range' if
    there aren't enough swings yet or they're mixed.
    """
    out = df_with_swings.copy()
    n = len(out)
    trend = ["range"] * n

    # Event queue: each swing (which occurred on row `idx`) only becomes
    # visible on row `confirmed_at_index` (not on row `idx` itself) — this
    # avoids lookahead without losing any past swing.
    events: list[tuple[int, str, int, float]] = []
    for j in range(n):
        if out.at[j, "swing_high"]:
            events.append((int(out.at[j, "confirmed_at_index"]), "high", j, out.at[j, "high"]))
        if out.at[j, "swing_low"]:
            events.append((int(out.at[j, "confirmed_at_index"]), "low", j, out.at[j, "low"]))
    events.sort(key=lambda e: e[0])

    swing_highs: list[tuple[int, float]] = []  # (index, value), already confirmed
    swing_lows: list[tuple[int, float]] = []
    current_trend = "range"
    event_ptr = 0

    for i in range(n):
        while event_ptr < len(events) and events[event_ptr][0] <= i:
            _, kind, idx, value = events[event_ptr]
            (swing_highs if kind == "high" else swing_lows).append((idx, value))
            event_ptr += 1

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_highs = swing_highs[-1][1] > swing_highs[-2][1]
            higher_lows = swing_lows[-1][1] > swing_lows[-2][1]
            lower_highs = swing_highs[-1][1] < swing_highs[-2][1]
            lower_lows = swing_lows[-1][1] < swing_lows[-2][1]

            if higher_highs and higher_lows:
                current_trend = "up"
            elif lower_highs and lower_lows:
                current_trend = "down"
            else:
                current_trend = "range"

        trend[i] = current_trend

    out["trend"] = trend
    return out


def last_swing_range(
    df_with_swings: pd.DataFrame, as_of_index: int
) -> tuple[float, float, str] | None:
    """Returns (swing_low_level, swing_high_level, direction) of the last
    swing confirmed up to `as_of_index` (inclusive), using only swings
    already confirmed at that point (no lookahead). `direction` is 'up' if
    the most recent swing is a low followed by a high (recent bullish
    move), or 'down' otherwise. Returns None if there isn't enough history.
    """
    visible = df_with_swings[df_with_swings["confirmed_at_index"] <= as_of_index]
    highs = visible[visible["swing_high"]]
    lows = visible[visible["swing_low"]]
    if highs.empty or lows.empty:
        return None

    last_high_idx = highs.index[-1]
    last_low_idx = lows.index[-1]

    if last_high_idx > last_low_idx:
        direction = "up"  # low -> high: last leg was bullish
    else:
        direction = "down"

    lo = float(df_with_swings.at[last_low_idx, "low"])
    hi = float(df_with_swings.at[last_high_idx, "high"])
    return lo, hi, direction
