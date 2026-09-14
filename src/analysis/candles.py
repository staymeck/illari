"""Candlestick pattern detection.

Per docs/PLAN.md ("Candlestick patterns... only considered combined with
structural context, never as a signal on their own"): these functions only
recognize the raw geometric pattern. Deciding whether a pattern matters — e.g.
requiring a hammer to appear at a support/Fibonacci confluence zone — is the
caller's responsibility (see src/backtest/engine.py).

Covers all 18 patterns in resources/18-Patrones-de-velas-que-debes-conocer.pdf
(single-candle: hammer/hanging man, inverted hammer/shooting star, the 4 doji
variants; two-candle: bullish/bearish engulfing, bullish/bearish harami,
piercing pattern, dark cloud cover; three-candle: morning/evening star, three
white soldiers/black crows). Several single-candle shapes are geometrically
identical to a pattern with the opposite meaning — e.g. hammer and hanging man
are the same shape — see bullish_reversal_pattern's docstring.
"""
from __future__ import annotations

import pandas as pd


def body(row: pd.Series) -> float:
    return abs(row["close"] - row["open"])


def candle_range(row: pd.Series) -> float:
    return row["high"] - row["low"]


def upper_wick(row: pd.Series) -> float:
    return row["high"] - max(row["close"], row["open"])


def lower_wick(row: pd.Series) -> float:
    return min(row["close"], row["open"]) - row["low"]


def is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def is_doji(row: pd.Series, body_to_range_max: float = 0.1) -> bool:
    """Body is a tiny fraction of the candle's total range — an indecision candle."""
    rng = candle_range(row)
    if rng == 0:
        return True
    return body(row) / rng <= body_to_range_max


def is_hammer(row: pd.Series, lower_wick_min_ratio: float = 2.0, upper_wick_max_ratio: float = 0.3) -> bool:
    """Small body near the top of the range, a lower wick at least
    `lower_wick_min_ratio` times the body, and little to no upper wick — a
    bullish-reversal candle when it appears after a decline (context is the
    caller's responsibility)."""
    b = body(row)
    if b == 0:
        return False
    return lower_wick(row) >= lower_wick_min_ratio * b and upper_wick(row) <= upper_wick_max_ratio * b


def is_shooting_star(row: pd.Series, upper_wick_min_ratio: float = 2.0, lower_wick_max_ratio: float = 0.3) -> bool:
    """Mirror image of the hammer: small body near the bottom of the range and
    a long upper wick — a bearish-reversal candle when it appears after an advance."""
    b = body(row)
    if b == 0:
        return False
    return upper_wick(row) >= upper_wick_min_ratio * b and lower_wick(row) <= lower_wick_max_ratio * b


def is_bullish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    """A bearish candle followed by a bullish candle whose body fully engulfs
    the previous candle's body."""
    if not (is_bearish(prev_row) and is_bullish(row)):
        return False
    return row["open"] <= prev_row["close"] and row["close"] >= prev_row["open"]


def is_bearish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    """A bullish candle followed by a bearish candle whose body fully engulfs
    the previous candle's body."""
    if not (is_bullish(prev_row) and is_bearish(row)):
        return False
    return row["open"] >= prev_row["close"] and row["close"] <= prev_row["open"]


# --- Doji variants (resources/18-Patrones-de-velas-que-debes-conocer.pdf) ---
# A doji is a tiny-body candle (see is_doji); these three classify it further
# by how its wicks are distributed. Plain doji and long-legged doji are pure
# indecision signals (no directional bias per the reference material) —
# gravestone and dragonfly are the doji-level extremes of the shooting-star/
# hammer shapes below, and DO carry a directional reading.


def is_gravestone_doji(row: pd.Series, body_to_range_max: float = 0.1, lower_wick_max_ratio: float = 0.1) -> bool:
    """Doji-level tiny body at the very bottom of the range, long upper
    wick, virtually no lower wick — "Lápida"/Bearish Doji."""
    if not is_doji(row, body_to_range_max):
        return False
    rng = candle_range(row)
    if rng == 0:
        return False
    return lower_wick(row) / rng <= lower_wick_max_ratio


def is_dragonfly_doji(row: pd.Series, body_to_range_max: float = 0.1, upper_wick_max_ratio: float = 0.1) -> bool:
    """Doji-level tiny body at the very top of the range, long lower wick,
    virtually no upper wick — "Libélula"/Bullish Doji."""
    if not is_doji(row, body_to_range_max):
        return False
    rng = candle_range(row)
    if rng == 0:
        return False
    return upper_wick(row) / rng <= upper_wick_max_ratio


def is_long_legged_doji(row: pd.Series, body_to_range_max: float = 0.1, wick_min_ratio: float = 0.3) -> bool:
    """Doji-level tiny body with BOTH wicks substantial — high volatility
    plus strong indecision, no directional bias."""
    if not is_doji(row, body_to_range_max):
        return False
    rng = candle_range(row)
    if rng == 0:
        return False
    return upper_wick(row) / rng >= wick_min_ratio and lower_wick(row) / rng >= wick_min_ratio


def doji_pattern(row: pd.Series) -> str | None:
    """Classifies a doji-level candle into its named variant (most specific
    first), or None if the candle isn't a doji at all."""
    if not is_doji(row):
        return None
    if is_long_legged_doji(row):
        return "long_legged_doji"
    if is_gravestone_doji(row):
        return "gravestone_doji"
    if is_dragonfly_doji(row):
        return "dragonfly_doji"
    return "doji"


# --- Two-candle patterns ---


def is_bullish_harami(prev_row: pd.Series, row: pd.Series) -> bool:
    """A large bearish candle followed by a small bullish candle whose body
    sits entirely INSIDE the previous candle's body — the reverse of an
    engulfing pattern."""
    if not (is_bearish(prev_row) and is_bullish(row)):
        return False
    return row["open"] >= prev_row["close"] and row["close"] <= prev_row["open"]


def is_bearish_harami(prev_row: pd.Series, row: pd.Series) -> bool:
    """Mirror image: a large bullish candle followed by a small bearish
    candle contained inside its body."""
    if not (is_bullish(prev_row) and is_bearish(row)):
        return False
    return row["open"] <= prev_row["close"] and row["close"] >= prev_row["open"]


def is_piercing_pattern(prev_row: pd.Series, row: pd.Series) -> bool:
    """A large bearish candle, a small gap-down open, then a bullish candle
    closing back up through at least the midpoint of the first candle's
    body (but not past its open — a full recovery would be a bullish
    engulfing instead)."""
    if not (is_bearish(prev_row) and is_bullish(row)):
        return False
    prev_midpoint = (prev_row["open"] + prev_row["close"]) / 2
    return row["open"] < prev_row["close"] and prev_midpoint < row["close"] < prev_row["open"]


def is_dark_cloud_cover(prev_row: pd.Series, row: pd.Series) -> bool:
    """Mirror image of the piercing pattern: a large bullish candle, a small
    gap-up open, then a bearish candle closing back down through at least
    the midpoint of the first candle's body."""
    if not (is_bullish(prev_row) and is_bearish(row)):
        return False
    prev_midpoint = (prev_row["open"] + prev_row["close"]) / 2
    return row["open"] > prev_row["close"] and prev_row["open"] < row["close"] < prev_midpoint


# --- Three-candle patterns ---


def is_morning_star(df: pd.DataFrame, idx: int) -> bool:
    """Large bearish candle, a small-bodied indecision candle, then a large
    bullish candle closing back up through at least the midpoint of the
    first candle's body."""
    if idx < 2:
        return False
    first, middle, last = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
    if not (is_bearish(first) and is_bullish(last)):
        return False
    if not (body(middle) < body(first) and body(middle) < body(last)):
        return False
    first_midpoint = (first["open"] + first["close"]) / 2
    return last["close"] > first_midpoint


def is_evening_star(df: pd.DataFrame, idx: int) -> bool:
    """Mirror image of the morning star: large bullish, small indecision,
    then a large bearish candle closing back down past the first candle's
    midpoint."""
    if idx < 2:
        return False
    first, middle, last = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
    if not (is_bullish(first) and is_bearish(last)):
        return False
    if not (body(middle) < body(first) and body(middle) < body(last)):
        return False
    first_midpoint = (first["open"] + first["close"]) / 2
    return last["close"] < first_midpoint


def is_three_white_soldiers(df: pd.DataFrame, idx: int) -> bool:
    """Three consecutive large-bodied bullish candles, each closing higher
    than the last."""
    if idx < 2:
        return False
    candles = [df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]]
    if not all(is_bullish(c) and not is_doji(c) for c in candles):
        return False
    return candles[0]["close"] < candles[1]["close"] < candles[2]["close"]


def is_three_black_crows(df: pd.DataFrame, idx: int) -> bool:
    """Mirror image: three consecutive large-bodied bearish candles, each
    closing lower than the last."""
    if idx < 2:
        return False
    candles = [df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]]
    if not all(is_bearish(c) and not is_doji(c) for c in candles):
        return False
    return candles[0]["close"] > candles[1]["close"] > candles[2]["close"]


def bullish_reversal_pattern(df: pd.DataFrame, idx: int) -> str | None:
    """Checks candle `idx` (and earlier candles where needed) for a known
    bullish reversal pattern, most-specific-first (3-candle, then 2-candle,
    then single-candle). Returns the pattern name, or None.

    Two of the single-candle shapes are shared, geometrically identical,
    with a bearish pattern in bearish_reversal_pattern — a small body with
    one long wick reads as "hammer" or "inverted hammer" here, and as
    "hanging man" or "shooting star" there; only the trend context the
    candle appears in (the caller's job, not this module's — see the module
    docstring) tells them apart.
    """
    row = df.iloc[idx]

    if is_morning_star(df, idx):
        return "morning_star"
    if is_three_white_soldiers(df, idx):
        return "three_white_soldiers"
    if idx > 0:
        prev = df.iloc[idx - 1]
        if is_piercing_pattern(prev, row):
            return "piercing_pattern"
        if is_bullish_engulfing(prev, row):
            return "bullish_engulfing"
        if is_bullish_harami(prev, row):
            return "bullish_harami"
    if is_dragonfly_doji(row):
        return "dragonfly_doji"
    if is_hammer(row):
        return "hammer"
    if is_shooting_star(row):
        return "inverted_hammer"
    return None


def bearish_reversal_pattern(df: pd.DataFrame, idx: int) -> str | None:
    """Mirror image of bullish_reversal_pattern — see its docstring for the
    shared-shape note (hammer/hanging man, inverted hammer/shooting star)."""
    row = df.iloc[idx]

    if is_evening_star(df, idx):
        return "evening_star"
    if is_three_black_crows(df, idx):
        return "three_black_crows"
    if idx > 0:
        prev = df.iloc[idx - 1]
        if is_dark_cloud_cover(prev, row):
            return "dark_cloud_cover"
        if is_bearish_engulfing(prev, row):
            return "bearish_engulfing"
        if is_bearish_harami(prev, row):
            return "bearish_harami"
    if is_gravestone_doji(row):
        return "gravestone_doji"
    if is_shooting_star(row):
        return "shooting_star"
    if is_hammer(row):
        return "hanging_man"
    return None
