"""Setup piece: price breaking out of a Wyckoff-style accumulation range
on above-average volume, AFTER the range itself showed net buy-side
absorption (Volume Spread Analysis) — see src.analysis.wyckoff for the
full reasoning. Unlike support_touch (confirms a pullback to an already-
established trend), this fires on the breakout itself, timed by what the
range's own volume behavior suggested was coming.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.volume import relative_volume
from src.analysis.wyckoff import accumulation_bias, find_trading_range
from src.strategies.registry import SETUPS
from src.strategies.types import EvalContext, SetupResult


@SETUPS.register("wyckoff_accumulation_breakout")
def wyckoff_accumulation_breakout(ctx: EvalContext, params: dict) -> SetupResult | None:
    lookback = params.get("lookback", 20)
    max_range_pct = params.get("max_range_pct", 8.0)
    volume_window = params.get("volume_window", 20)
    volume_threshold = params.get("volume_threshold", 1.5)
    close_position_threshold = params.get("close_position_threshold", 0.3)
    min_accumulation_bias = params.get("min_accumulation_bias", 0.15)
    breakout_volume_mult = params.get("breakout_volume_mult", 1.2)

    df = ctx.price_window
    range_info = find_trading_range(df, lookback=lookback, max_range_pct=max_range_pct)
    if range_info is None:
        return None

    bias = accumulation_bias(
        df, range_info, volume_window=volume_window,
        volume_threshold=volume_threshold, close_position_threshold=close_position_threshold,
    )
    if bias < min_accumulation_bias:
        return None

    current_close = float(df["close"].iloc[-1])
    if current_close <= range_info["high"]:
        return None  # hasn't actually broken out yet

    breakout_rel_vol = relative_volume(df, window=volume_window).iloc[-1]
    if pd.isna(breakout_rel_vol) or breakout_rel_vol < breakout_volume_mult:
        return None  # broke out, but not on real participation - could be a fakeout

    return SetupResult(
        reference_level=range_info["high"],
        extras={
            "range_low": range_info["low"],
            "range_high": range_info["high"],
            "accumulation_bias": round(bias, 3),
        },
    )
