"""Top-down confluence strategy: daily bias -> 1h zone/structure -> 5m
entry trigger (candle pattern + volume).

A signal is only generated when all 4 conditions line up:
  1. Daily bias (1D) aligned with the proposed direction.
  2. Entry candle's price inside the 1h structure's Fibonacci zone
     (0.5-0.618 of the last swing by default).
  3. Confirmation candle pattern on the entry candle (5m).
  4. Entry candle's volume not anomalously low.

Optionally, a 5th condition: `strategy.trendline_required=True` also
requires the price to be close (within `trendline_proximity_atr_mult *
ATR`) to the least-squares support/resistance line (analytic geometry, see
`indicators/trendline.py`) — disabled by default, where proximity to the
line only adds a bonus to the score (`trendline_bonus_enabled`).

The confidence score sums the mandatory conditions plus bonuses for
confluence quality (strong volume, a more decisive pattern like engulfing,
proximity to the line if it wasn't required as a condition). All the
per-entry-candle computation only uses 1h/1D data already closed at that
point (see `data/mtf_align.py`).
"""

from __future__ import annotations

import pandas as pd

from trading_lab.data.mtf_align import align_multi_timeframe
from trading_lab.indicators import candles, fibonacci, session, structure, trendline, volatility, volume
from trading_lab.strategy.base import Signal, Strategy


def _swing_zone_for_row(
    out: pd.DataFrame, i: int, fib_zone_min: float, fib_zone_max: float
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Fibonacci zone (and the swing that produces it) for row `i`, based
    on the last swing high and last swing low confirmed up to that row.
    `find_swings` doesn't force highs and lows to alternate, so the last
    confirmed high and the last confirmed low aren't always consecutive —
    if the most recent low ends up at a price >= the most recent high,
    they don't form a valid retracement range and it's treated as "no
    zone" instead of forcing an invalid computation (avoids the
    ValueError from `compute_fib_levels`).
    """
    rng = structure.last_swing_range(out, as_of_index=i)
    if rng is None or rng[1] <= rng[0]:
        return None, None, None, None, None

    lo, hi, direction = rng
    levels = fibonacci.compute_fib_levels(lo, hi, direction)
    zlo, zhi = fibonacci.confluence_zone(levels, fib_zone_min, fib_zone_max)
    return zlo, zhi, lo, hi, direction


def prepare_structure_df(
    df: pd.DataFrame,
    swing_lookback: int,
    fib_zone_min: float,
    fib_zone_max: float,
    trendline_min_points: int = 3,
    trendline_max_points: int = 6,
) -> pd.DataFrame:
    """Computes swings, trend, the last swing's Fibonacci zone, and the
    least-squares support/resistance lines (analytic geometry) over recent
    swings — candle by candle, on the structure timeframe (1h)."""
    out = structure.find_swings(df, lookback=swing_lookback)
    out = structure.trend_from_swings(out)
    out = trendline.attach_trendline_features(out, trendline_min_points, trendline_max_points)

    zone_lo, zone_hi, swing_low, swing_high, swing_dir = [], [], [], [], []
    for i in range(len(out)):
        zlo, zhi, lo, hi, direction = _swing_zone_for_row(out, i, fib_zone_min, fib_zone_max)
        zone_lo.append(zlo)
        zone_hi.append(zhi)
        swing_low.append(lo)
        swing_high.append(hi)
        swing_dir.append(direction)

    out["zone_lo"] = zone_lo
    out["zone_hi"] = zone_hi
    out["swing_low_level"] = swing_low
    out["swing_high_level"] = swing_high
    out["swing_direction"] = swing_dir
    return out


def prepare_bias_df(df: pd.DataFrame, swing_lookback: int) -> pd.DataFrame:
    """Computes the (bias) trend candle by candle on the daily timeframe."""
    out = structure.find_swings(df, lookback=swing_lookback)
    out = structure.trend_from_swings(out)
    return out


def prepare_entry_df(df: pd.DataFrame, volume_lookback: int, volume_min_ratio: float, atr_period: int) -> pd.DataFrame:
    """Computes candle patterns, volume sufficiency, ATR, and hour/session
    on the entry timeframe (5m)."""
    out = df.copy()
    out["confirm_bullish"] = candles.bullish_confirmation(out)
    out["confirm_bearish"] = candles.bearish_confirmation(out)
    out["is_engulfing_bull"] = candles.is_bullish_engulfing(out)
    out["is_engulfing_bear"] = candles.is_bearish_engulfing(out)
    out["volume_ok"] = volume.is_volume_sufficient(out, volume_lookback, volume_min_ratio)
    out["volume_ratio"] = volume.volume_ratio(out, volume_lookback)
    out["atr"] = volatility.atr(out, atr_period)
    out = session.add_session_columns(out)
    return out


def prepare_merged(
    entry_df: pd.DataFrame, structure_df: pd.DataFrame, bias_df: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Prepares the 3 timeframes (each one's own indicators) and aligns
    them with no lookahead. Used both to generate signals and for the
    per-ingredient probability analysis (`analysis/probability.py`), which
    needs the same already-built multi-timeframe DataFrame."""
    entry_tf = params["timeframes"]["entry"]
    structure_tf = params["timeframes"]["structure"]
    bias_tf = params["timeframes"]["bias"]
    sp = params["strategy"]

    prepared_entry = prepare_entry_df(entry_df, sp["volume_lookback"], sp["volume_min_ratio"], sp["atr_period"])
    prepared_structure = prepare_structure_df(
        structure_df,
        sp["swing_lookback"],
        sp["fib_zone_min"],
        sp["fib_zone_max"],
        sp.get("trendline_min_points", 3),
        sp.get("trendline_max_points", 6),
    )
    prepared_bias = prepare_bias_df(bias_df, sp["swing_lookback"])

    return align_multi_timeframe(prepared_entry, entry_tf, prepared_structure, structure_tf, prepared_bias, bias_tf)


class ConfluenceStrategy(Strategy):
    def generate_signals(
        self,
        entry_df: pd.DataFrame,
        structure_df: pd.DataFrame,
        bias_df: pd.DataFrame,
        params: dict,
    ) -> list[Signal]:
        sp = params["strategy"]
        merged = prepare_merged(entry_df, structure_df, bias_df, params)

        signals: list[Signal] = []
        for _, row in merged.iterrows():
            sig = self._evaluate_row(row, sp)
            if sig is not None:
                signals.append(sig)
        return signals

    @staticmethod
    def _evaluate_row(row: pd.Series, sp: dict) -> Signal | None:
        bias_trend = row.get("bias_trend")
        struct_trend = row.get("struct_trend")
        zone_lo, zone_hi = row.get("struct_zone_lo"), row.get("struct_zone_hi")
        swing_low, swing_high = row.get("struct_swing_low_level"), row.get("struct_swing_high_level")
        close = row["close"]

        if bias_trend not in ("up", "down") or struct_trend not in ("up", "down"):
            return None
        if pd.isna(zone_lo) or pd.isna(zone_hi):
            return None
        if not (zone_lo <= close <= zone_hi):
            return None
        if not row.get("volume_ok", False):
            return None

        direction = None
        if bias_trend == "up" and struct_trend == "up" and row.get("confirm_bullish", False):
            direction = "long"
        elif bias_trend == "down" and struct_trend == "down" and row.get("confirm_bearish", False):
            direction = "short"

        if direction is None:
            return None

        atr = row.get("atr") or 0.0
        if direction == "long":
            structural_stop = swing_low if pd.notna(swing_low) else None
            use_structural = structural_stop is not None and structural_stop < close
            stop_loss = structural_stop if use_structural else close - sp["atr_sl_multiplier"] * atr
            risk = close - stop_loss
            take_profit = close + sp["reward_risk_ratio"] * risk
        else:
            structural_stop = swing_high if pd.notna(swing_high) else None
            use_structural = structural_stop is not None and structural_stop > close
            stop_loss = structural_stop if use_structural else close + sp["atr_sl_multiplier"] * atr
            risk = stop_loss - close
            take_profit = close - sp["reward_risk_ratio"] * risk

        if risk <= 0:
            return None

        # Analytic geometry: is the price close (within
        # trendline_proximity_atr_mult * ATR) to the least-squares support
        # (long) or resistance (short) line over structure swings? Computed
        # once and used both for the mandatory filter (if enabled) and for
        # the bonus.
        proximity_mult = sp.get("trendline_proximity_atr_mult", 1.0)
        dist_col = "struct_dist_to_support" if direction == "long" else "struct_dist_to_resistance"
        dist = row.get(dist_col)
        near_trendline = pd.notna(dist) and atr > 0 and abs(dist) <= proximity_mult * atr

        if sp.get("trendline_required", False) and not near_trendline:
            return None

        confidence = 4  # the 4 mandatory conditions
        reasons = {
            "bias_trend": bias_trend,
            "structure_trend": struct_trend,
            "zone": [round(zone_lo, 2), round(zone_hi, 2)],
            "volume_ratio": round(float(row.get("volume_ratio", 1.0)), 2),
        }
        if sp.get("trendline_required", False):
            confidence += 1  # geometry became the 5th mandatory condition
            reasons["geometry_required"] = True

        if row.get("volume_ratio", 1.0) and row["volume_ratio"] > 1.5:
            confidence += 1
            reasons["volume_bonus"] = True
        engulfing = row.get("is_engulfing_bull", False) or row.get("is_engulfing_bear", False)
        if engulfing:
            confidence += 1
            reasons["pattern"] = "engulfing"
        else:
            reasons["pattern"] = "pin_bar_or_hammer"

        # If geometry was already required as a mandatory condition, it
        # doesn't make sense to add it again as an informational bonus.
        if sp.get("trendline_bonus_enabled", True) and not sp.get("trendline_required", False) and near_trendline:
            confidence += 1
            reasons["geometry_bonus"] = True

        return Signal(
            timestamp=row["timestamp"],
            direction=direction,
            entry_price=float(close),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence_score=confidence,
            hour_utc=int(row["hour_utc"]),
            session=row["session"],
            reasons=reasons,
        )
