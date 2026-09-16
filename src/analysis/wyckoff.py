"""Wyckoff-style accumulation/distribution range detection and Volume
Spread Analysis (VSA) — a genuinely different entry-timing idea from the
rest of this catalog: instead of confirming a trend already under way
(structure + Fibonacci + candle pattern, all AFTER price has moved),
this looks for a SIDEWAYS range where volume suggests a large participant
is absorbing supply (buying without letting price fall much) or demand
(selling without letting price rise much) — the "cause" in Wyckoff's own
language — and projects a target from the range's own height once price
breaks out — the "effect", the classic Wyckoff measured-move.

Motivated directly by the user's own framing: candle-pattern confirmation
answers "did a move already start", not "is the market accumulating
liquidity right now, and how far could it go once it lets go" — this
module is aimed at the second question instead.

Deliberately scoped to what OHLCV + volume can actually support: no
order-book depth, no true trade-by-trade order flow (Binance spot's
public API offers neither for free — see docs/PLAN.md's feasibility
note). This is classical Wyckoff (Richard Wyckoff, early 1900s) / VSA
(Tom Williams, "Master the Markets", 1990s) — every signal below has an
explicit, objective, backtestable definition, unlike "Smart Money
Concepts"/ICT-style liquidity-grab reading, which has none.

Reuses src.analysis.volume.relative_volume (already no-lookahead via
shift(1)) rather than re-deriving volume normalization here.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.volume import relative_volume


def find_trading_range(df: pd.DataFrame, lookback: int = 20, max_range_pct: float = 8.0) -> dict | None:
    """Looks at the `lookback` bars immediately BEFORE the most recent one
    (the most recent bar is left out — it's the candidate breakout bar,
    not part of the range being measured) and checks whether price stayed
    within a tight band. A trading range exists when
    (range_high - range_low) / range_low <= max_range_pct — a plain,
    objective, purely price-based definition (no subjective pattern
    reading). Returns None if there isn't enough history or the recent
    bars were too volatile to call a range."""
    if len(df) < lookback + 1:
        return None
    window = df.iloc[-(lookback + 1) : -1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    if range_low <= 0:
        return None
    range_pct = (range_high - range_low) / range_low * 100
    if range_pct > max_range_pct:
        return None
    return {"high": range_high, "low": range_low, "n_bars": len(window)}


def close_position(df: pd.DataFrame) -> pd.Series:
    """Where each bar closed within its own high-low range: 0 = closed at
    the low, 1 = closed at the high, 0.5 = closed in the middle. A doji
    (zero-range bar) is treated as a neutral 0.5 close, not NaN."""
    spread = (df["high"] - df["low"]).replace(0, float("nan"))
    return ((df["close"] - df["low"]) / spread).fillna(0.5)


def accumulation_bias(
    df: pd.DataFrame,
    range_info: dict,
    volume_window: int = 20,
    volume_threshold: float = 1.5,
    close_position_threshold: float = 0.3,
) -> float:
    """Net Volume Spread Analysis bias within the range's own bars: +1 per
    bullish-absorption bar (above-average volume — real effort — that
    still closed in the upper `close_position_threshold` of its own
    range, i.e. buying held price up despite the volume: "stopping
    volume") and -1 per bearish-absorption bar (closed in the lower
    `close_position_threshold`), net and normalized by bar count.
    Positive = net buy-side absorption (accumulation bias), negative =
    net sell-side (distribution bias), 0 = no clear bias or nothing
    detected."""
    n_bars = range_info["n_bars"]
    window = df.iloc[-(n_bars + 1) : -1]
    rel_vol = relative_volume(df, window=volume_window).iloc[-(n_bars + 1) : -1]
    pos = close_position(window)

    high_effort = rel_vol >= volume_threshold
    bullish_absorption = high_effort & (pos >= 1 - close_position_threshold)
    bearish_absorption = high_effort & (pos <= close_position_threshold)

    net = int(bullish_absorption.sum()) - int(bearish_absorption.sum())
    return net / n_bars if n_bars else 0.0


def range_projected_target(entry_price: float, range_high: float, range_low: float, direction: str = "long") -> float:
    """The classic Wyckoff measured-move: the trading range's own height
    is the "cause", projected from the breakout point as the expected
    "effect". `direction="long"` projects upward from `entry_price`;
    `"short"` projects downward. Deliberately the simplest, most
    defensible version of Wyckoff's target-projection idea — his own
    Point & Figure counting method needs tick-by-tick figure charts this
    project doesn't build."""
    height = range_high - range_low
    if direction == "long":
        return entry_price + height
    if direction == "short":
        return entry_price - height
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
