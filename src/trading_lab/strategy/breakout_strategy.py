"""Donchian Channel Breakout ("Turtle Trading", Richard Dennis — 1980s):
a classic, widely documented trend-following approach. Offered as a second,
independent strategy (same `Strategy` interface, same backtest engine,
same probability analysis) to compare head to head against the confluence
strategy — instead of continuing to iterate on a single family of rules.

Long entry: the close breaks above the highest high of the `donchian_period_bars`
PREVIOUS candles (with `.shift(1)`, excluding the current candle — no
lookahead) AND the daily bias is 'up'. Short entry: breaks below the
lowest low AND the daily bias is 'down'. Same top-down filter as
`confluence_strategy.py`, to keep the comparison fair.

`fade` (default `false`): FLIPS the direction — instead of following the
breakout, it trades against it (betting on a reversal). This was added
after the probability analysis, with 36 months of data, showed that this
strategy's breakout signals have a statistically significant NEGATIVE
edge (p≈0 in both scenarios) — the market reverses more often than it
continues after breaking the 5m channel. `fade` trades exactly on that
evidence: same entry conditions, flipped direction.

`require_bias_alignment` (default `true`): if `false`, the daily bias is
ignored entirely — any breakout counts (with or without `fade`), not only
ones aligned with the day's trend. Used to compare whether the daily bias
was actually helping or diluting the signal.

`dynamic_stop_enabled` / `dynamic_risk_enabled` (both default `false`):
same mechanism as in `confluence_strategy.py` — scale the stop distance /
position size by the current volatility regime (current ATR vs. its own
recent average) instead of a fixed `atr_sl_multiplier` and a fixed
`risk_per_trade_pct`. This strategy is where the gap mattered most: the
probability analysis found a statistically significant directional edge in
`fade`'s signals (p≈0, hit rate 57-62% vs. a ~50% base rate over a 1h
horizon), yet every `fade` variant still lost almost all its capital when
actually traded — because the fixed 1.5xATR stop, placed right at the
breakout (the single noisiest moment for this setup), got hit by normal
retracement before the validated 1h move had time to develop. `fade`
combined with `dynamic_stop_enabled` targets exactly that mismatch.

Three OPTIONAL filters (disabled by default, so already-run variants keep
their behavior), specifically aimed at over-trading on short timeframes:
  - `min_channel_width_atr_mult`: only counts a breakout if the channel
    that was broken is "wide enough" (>= this multiple of ATR) — a narrow
    channel on 5m candles is usually noise, not real structure.
  - `volatility_expansion_required`: only counts a breakout if the current
    ATR is ABOVE its own recent moving average — i.e. if volatility is
    expanding (the breakout has "force" behind it), not if the market is
    still flat. This is a filter *coincident* with the breakout (it
    measures expansion already under way).
  - `narrow_range_required`: an *anticipatory* filter (Toby Crabel, "Day
    Trading with Short Term Price Patterns") — requires the candle
    IMMEDIATELY BEFORE the breakout to have been narrow range (NR7, see
    `indicators/candle_geometry.py`): "compression before the move",
    instead of measuring expansion already under way.
"""

from __future__ import annotations

import pandas as pd

from trading_lab.data.mtf_align import align_higher_timeframe
from trading_lab.indicators import candle_geometry, session, volatility, volume
from trading_lab.strategy.base import Signal, Strategy
from trading_lab.strategy.confluence_strategy import prepare_bias_df


def prepare_entry_df(
    df: pd.DataFrame,
    donchian_period_bars: int,
    atr_period: int,
    volume_lookback: int,
    atr_expansion_lookback: int = 20,
    narrow_range_lookback: int = 7,
) -> pd.DataFrame:
    """Donchian channel, ATR (+ its own moving average, for the volatility
    expansion filter), channel width, candle geometry (NR7), relative
    volume, and hour/session on the entry timeframe (5m)."""
    out = df.copy()
    # .shift(1): the channel is computed from candles ALREADY closed before
    # the current one — the candle that breaks the channel can't be part
    # of it (otherwise "breaking its own high" would be trivially true).
    out["donchian_high"] = (
        out["high"].rolling(window=donchian_period_bars, min_periods=donchian_period_bars).max().shift(1)
    )
    out["donchian_low"] = (
        out["low"].rolling(window=donchian_period_bars, min_periods=donchian_period_bars).min().shift(1)
    )
    out["breakout_up"] = out["close"] > out["donchian_high"]
    out["breakout_down"] = out["close"] < out["donchian_low"]
    out["channel_width"] = out["donchian_high"] - out["donchian_low"]
    out["atr"] = volatility.atr(out, atr_period)
    out["atr_sma"] = out["atr"].rolling(window=atr_expansion_lookback, min_periods=1).mean()
    # Anticipatory filter (Crabel): the candle BEFORE the current one was
    # narrow range (NR7) -> "compression before the move". .shift(1) so we
    # don't look at the breakout candle itself (which is typically wide).
    out["prior_bar_narrow_range"] = candle_geometry.is_narrow_range(out, narrow_range_lookback).shift(1)
    out["volume_ratio"] = volume.volume_ratio(out, volume_lookback)
    out = session.add_session_columns(out)
    return out


def prepare_merged(entry_df: pd.DataFrame, bias_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Prepares entry (Donchian/ATR/volume) + daily bias and aligns them
    with no lookahead. Doesn't use the structure timeframe (1h) — Donchian
    only needs the entry and the daily bias."""
    sp = params["strategy"]
    entry_tf = params["timeframes"]["entry"]
    bias_tf = params["timeframes"]["bias"]

    prepared_entry = prepare_entry_df(
        entry_df,
        sp.get("donchian_period_bars", 96),
        sp["atr_period"],
        sp["volume_lookback"],
        sp.get("atr_expansion_lookback", 20),
        sp.get("narrow_range_lookback", 7),
    )
    prepared_bias = prepare_bias_df(bias_df, sp["swing_lookback"])
    return align_higher_timeframe(prepared_entry, entry_tf, prepared_bias, bias_tf, prefix="bias")


class DonchianBreakoutStrategy(Strategy):
    def generate_signals(
        self,
        entry_df: pd.DataFrame,
        structure_df: pd.DataFrame,  # unused: Donchian only needs entry + daily bias
        bias_df: pd.DataFrame,
        params: dict,
    ) -> list[Signal]:
        del structure_df  # part of the common `Strategy` signature, not used here
        sp = params["strategy"]
        merged = prepare_merged(entry_df, bias_df, params)

        signals: list[Signal] = []
        for _, row in merged.iterrows():
            sig = self._evaluate_row(row, sp)
            if sig is not None:
                signals.append(sig)
        return signals

    @staticmethod
    def _evaluate_row(row: pd.Series, sp: dict) -> Signal | None:
        bias_trend = row.get("bias_trend")
        require_bias = sp.get("require_bias_alignment", True)
        if require_bias and bias_trend not in ("up", "down"):
            return None

        fade = sp.get("fade", False)
        close = row["close"]
        direction = None

        if require_bias:
            if row.get("breakout_up", False) and bias_trend == "up":
                direction = "short" if fade else "long"
            elif row.get("breakout_down", False) and bias_trend == "down":
                direction = "long" if fade else "short"
        else:
            if row.get("breakout_up", False):
                direction = "short" if fade else "long"
            elif row.get("breakout_down", False):
                direction = "long" if fade else "short"

        if direction is None:
            return None

        atr = row.get("atr") or 0.0
        if not atr or atr <= 0:
            return None

        min_width_mult = sp.get("min_channel_width_atr_mult", 0.0)
        if min_width_mult > 0:
            channel_width = row.get("channel_width")
            if pd.isna(channel_width) or channel_width < min_width_mult * atr:
                return None

        if sp.get("volatility_expansion_required", False):
            atr_sma = row.get("atr_sma")
            if pd.isna(atr_sma) or atr <= atr_sma:
                return None

        if sp.get("narrow_range_required", False):
            if row.get("prior_bar_narrow_range") is not True:
                return None

        # Dynamic stop/sizing: same volatility-regime mechanism as
        # confluence_strategy.py, reusing the atr_sma already computed for
        # the (coincident) volatility_expansion_required filter.
        atr_sma = row.get("atr_sma")
        regime_ratio = (atr / atr_sma) if (atr_sma and atr_sma > 0) else None

        sl_multiplier = sp["atr_sl_multiplier"]
        if sp.get("dynamic_stop_enabled", False):
            regime = volatility.regime_multiplier_scalar(
                regime_ratio, sp.get("dynamic_stop_min_mult", 0.75), sp.get("dynamic_stop_max_mult", 2.0)
            )
            sl_multiplier = sl_multiplier * regime

        if direction == "long":
            stop_loss = close - sl_multiplier * atr
            risk = close - stop_loss
            take_profit = close + sp["reward_risk_ratio"] * risk
        else:
            stop_loss = close + sl_multiplier * atr
            risk = stop_loss - close
            take_profit = close - sp["reward_risk_ratio"] * risk

        if risk <= 0:
            return None

        risk_multiplier = 1.0
        if sp.get("dynamic_risk_enabled", False):
            regime = volatility.regime_multiplier_scalar(
                regime_ratio, sp.get("dynamic_stop_min_mult", 0.75), sp.get("dynamic_stop_max_mult", 2.0)
            )
            risk_multiplier = 1.0 / regime if regime > 0 else 1.0

        confidence = 2  # channel breakout (+ daily bias aligned, if `require_bias_alignment`)
        reasons = {"bias_trend": bias_trend, "breakout": direction}
        if fade:
            reasons["fade"] = True
        if not require_bias:
            reasons["require_bias_alignment"] = False
        if min_width_mult > 0:
            reasons["min_channel_width_atr_mult"] = min_width_mult
        if sp.get("volatility_expansion_required", False):
            reasons["volatility_expansion_required"] = True
        if sp.get("narrow_range_required", False):
            reasons["narrow_range_required"] = True
        if sp.get("dynamic_stop_enabled", False):
            reasons["dynamic_stop"] = True
        if sp.get("dynamic_risk_enabled", False):
            reasons["dynamic_risk"] = True

        volume_ratio = row.get("volume_ratio")
        if volume_ratio is not None and pd.notna(volume_ratio) and volume_ratio > 1.2:
            # Breakout "on volume" — classic breakout quality criterion.
            confidence += 1
            reasons["volume_bonus"] = True

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
            risk_multiplier=float(risk_multiplier),
        )
