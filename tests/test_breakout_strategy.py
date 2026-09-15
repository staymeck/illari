"""Integration test: builds synthetic entry_df/bias_df, hand-designed to
trigger exactly ONE known Donchian channel breakout, and verifies that
`DonchianBreakoutStrategy` detects it with the expected direction and risk
management."""

import pandas as pd
import pytest

from trading_lab.strategy.breakout_strategy import DonchianBreakoutStrategy

# Same zigzag as in test_confluence_strategy.py: with swing_lookback=1,
# confirms an 'up' trend from index 5 onward (day 2024-01-06 onward).
ZIGZAG = [100.0, 90.0, 105.0, 95.0, 115.0, 110.0, 130.0]

PARAMS = {
    "timeframes": {"entry": "5m", "structure": "1h", "bias": "1d"},
    "strategy": {
        "donchian_period_bars": 3,
        "swing_lookback": 1,
        "volume_lookback": 3,
        "atr_period": 3,
        "atr_sl_multiplier": 1.5,
        "reward_risk_ratio": 2.0,
        "risk_per_trade_pct": 1.0,
        "commission_pct": 0.1,
        "slippage_pct": 0.05,
        "probability_horizon_bars": 12,
    },
}


def _bias_df() -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=7, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": list(days),
            "open": ZIGZAG,
            "high": ZIGZAG,
            "low": ZIGZAG,
            "close": ZIGZAG,
            "volume": [100.0] * len(ZIGZAG),
        }
    )


def _entry_df() -> pd.DataFrame:
    # The 'up' daily bias is confirmed from 2024-01-06 (closes at 00:00 on
    # the 7th) -> every candle from the 7th onward already sees it.
    start = pd.Timestamp("2024-01-07T00:00:00Z")
    timestamps = [start + pd.Timedelta(minutes=5 * i) for i in range(5)]

    rows = [
        # open,  high,  low,   close, volume
        (97.0, 100.0, 95.0, 98.0, 100.0),
        (98.0, 101.0, 96.0, 99.0, 100.0),
        (99.0, 102.0, 97.0, 100.0, 100.0),
        (101.0, 108.0, 100.0, 107.0, 300.0),  # BREAKOUT: closes above the max of the 3 previous candles (102)
        (107.0, 109.0, 106.0, 108.0, 100.0),
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = timestamps
    return df


def test_breakout_strategy_detects_single_expected_long_signal():
    entry_df, bias_df = _entry_df(), _bias_df()
    empty_structure_df = pd.DataFrame()  # Donchian doesn't use 1h structure

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, empty_structure_df, bias_df, PARAMS)

    assert len(signals) == 1
    sig = signals[0]

    assert sig.direction == "long"
    assert sig.timestamp == pd.Timestamp("2024-01-07T00:15:00Z")
    assert sig.entry_price == pytest.approx(107.0)
    assert sig.stop_loss < sig.entry_price < sig.take_profit
    assert sig.hour_utc == 0
    assert sig.session == "asia"
    # Breakout candle's volume well above average -> bonus.
    assert sig.confidence_score == 3
    assert sig.reasons["volume_bonus"] is True


def test_breakout_strategy_no_signal_without_daily_bias():
    entry_df, bias_df = _entry_df(), _bias_df()
    bias_df = bias_df.iloc[:5]  # cut before the bias confirms 'up'
    empty_structure_df = pd.DataFrame()

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, empty_structure_df, bias_df, PARAMS)

    assert signals == []


def test_fade_flips_direction_of_known_signal():
    entry_df, bias_df = _entry_df(), _bias_df()
    params = {**PARAMS, "strategy": {**PARAMS["strategy"], "fade": True}}

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.direction == "short"  # the detected breakout was still bullish; fade flips it
    assert sig.entry_price == pytest.approx(107.0)
    assert sig.take_profit < sig.entry_price < sig.stop_loss  # inverted risk for a short
    assert sig.reasons["fade"] is True


def test_require_bias_alignment_false_with_fade_ignores_daily_bias():
    entry_df, bias_df = _entry_df(), _bias_df()
    bias_df = bias_df.iloc[:5]  # bias never confirms 'up' -> normally, no signal
    params = {
        **PARAMS,
        "strategy": {**PARAMS["strategy"], "fade": True, "require_bias_alignment": False},
    }

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    assert signals[0].direction == "short"
    assert signals[0].reasons["require_bias_alignment"] is False


def test_require_bias_alignment_false_without_fade_still_follows_breakout():
    entry_df, bias_df = _entry_df(), _bias_df()
    bias_df = bias_df.iloc[:5]  # bias never confirms 'up'
    params = {
        **PARAMS,
        "strategy": {**PARAMS["strategy"], "require_bias_alignment": False},
    }

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    assert signals[0].direction == "long"  # without fade, it follows the breakout as usual


def test_min_channel_width_filter_blocks_narrow_breakout():
    # On the breakout candle: channel_width=7.0, atr=6.0 -> ratio ~1.17.
    # Requiring >= 1.5x ATR (9.0) should discard the signal.
    entry_df, bias_df = _entry_df(), _bias_df()
    params = {**PARAMS, "strategy": {**PARAMS["strategy"], "min_channel_width_atr_mult": 1.5}}

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert signals == []


def test_min_channel_width_filter_allows_wide_enough_breakout():
    # Same case, but requiring only >= 1.0x ATR (6.0) -> 7.0 is enough.
    entry_df, bias_df = _entry_df(), _bias_df()
    params = {**PARAMS, "strategy": {**PARAMS["strategy"], "min_channel_width_atr_mult": 1.0}}

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    assert signals[0].reasons["min_channel_width_atr_mult"] == 1.0


def test_volatility_expansion_filter_allows_genuine_expansion():
    # On the breakout candle, atr(6.0) > atr_sma(5.25) -> volatility is
    # genuinely expanding -> the filter shouldn't block this signal.
    entry_df, bias_df = _entry_df(), _bias_df()
    params = {**PARAMS, "strategy": {**PARAMS["strategy"], "volatility_expansion_required": True}}

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    assert signals[0].reasons["volatility_expansion_required"] is True


def test_narrow_range_filter_allows_breakout_with_prior_compression():
    # With lookback=3, candle 2 (immediately before the breakout on candle
    # 3) is the narrowest-range one of [candle0,candle1,candle2] (all
    # three have range 5, there's a tie -> counts as narrow) -> the filter
    # shouldn't block the known signal.
    entry_df, bias_df = _entry_df(), _bias_df()
    params = {
        **PARAMS,
        "strategy": {**PARAMS["strategy"], "narrow_range_required": True, "narrow_range_lookback": 3},
    }

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert len(signals) == 1
    assert signals[0].reasons["narrow_range_required"] is True


def test_narrow_range_filter_blocks_breakout_without_prior_compression():
    # Narrow candle 1 (range 4) so that candle 2 (immediately before the
    # breakout, range 5) stops being the narrowest one in its window
    # [candle0=5, candle1=4, candle2=5] -> the filter should discard the signal.
    entry_df, bias_df = _entry_df(), _bias_df()
    entry_df.loc[1, "high"] = 100.5
    entry_df.loc[1, "low"] = 96.5
    params = {
        **PARAMS,
        "strategy": {**PARAMS["strategy"], "narrow_range_required": True, "narrow_range_lookback": 3},
    }

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, pd.DataFrame(), bias_df, params)

    assert signals == []


def _minimal_long_breakout_row(**overrides) -> pd.Series:
    """A row with just enough fields to reach `_evaluate_row`'s stop/sizing
    logic directly, for testing dynamic stop/sizing in isolation from the
    rest of the breakout conditions."""
    base = {
        "timestamp": pd.Timestamp("2024-01-07T00:15:00Z"),
        "bias_trend": "up",
        "breakout_up": True,
        "breakout_down": False,
        "close": 100.0,
        "atr": 10.0,
        "atr_sma": 5.0,  # atr(10) / atr_sma(5) = regime ratio of 2.0
        "channel_width": float("nan"),
        "prior_bar_narrow_range": None,
        "volume_ratio": 1.0,
        "hour_utc": 0,
        "session": "asia",
    }
    base.update(overrides)
    return pd.Series(base)


def test_breakout_dynamic_stop_widens_with_expanded_volatility_regime():
    sp = {**PARAMS["strategy"], "dynamic_stop_enabled": True}
    row = _minimal_long_breakout_row()  # atr/atr_sma = 2.0

    sig = DonchianBreakoutStrategy._evaluate_row(row, sp)

    # stop = close - (atr_sl_multiplier * regime) * atr = 100 - (1.5*2.0)*10 = 70.
    assert sig.stop_loss == pytest.approx(100.0 - 1.5 * 2.0 * 10.0)
    assert sig.reasons["dynamic_stop"] is True


def test_breakout_dynamic_stop_disabled_ignores_volatility_regime():
    sp = {**PARAMS["strategy"], "dynamic_stop_enabled": False}
    row = _minimal_long_breakout_row()

    sig = DonchianBreakoutStrategy._evaluate_row(row, sp)

    assert sig.stop_loss == pytest.approx(100.0 - 1.5 * 10.0)
    assert "dynamic_stop" not in sig.reasons


def test_breakout_dynamic_risk_reduces_position_risk_when_volatility_expanded():
    sp = {**PARAMS["strategy"], "dynamic_risk_enabled": True}
    row = _minimal_long_breakout_row()

    sig = DonchianBreakoutStrategy._evaluate_row(row, sp)

    assert sig.risk_multiplier == pytest.approx(0.5)  # 1 / regime(2.0)
    assert sig.reasons["dynamic_risk"] is True


def test_breakout_fade_combined_with_dynamic_stop():
    # The combination the probability analysis actually pointed to: fade's
    # direction has a real edge, dynamic_stop targets the fixed-stop/noise
    # mismatch that was destroying it in execution.
    sp = {**PARAMS["strategy"], "fade": True, "dynamic_stop_enabled": True}
    row = _minimal_long_breakout_row()  # detected breakout is still bullish; fade flips it

    sig = DonchianBreakoutStrategy._evaluate_row(row, sp)

    assert sig.direction == "short"
    # Short stop = close + (atr_sl_multiplier * regime) * atr = 100 + 30 = 130.
    assert sig.stop_loss == pytest.approx(100.0 + 1.5 * 2.0 * 10.0)
    assert sig.reasons["fade"] is True
    assert sig.reasons["dynamic_stop"] is True


def test_breakout_strategy_no_signal_without_channel_break():
    # Without the breakout candle (dataset cut short before it), the
    # channel is never exceeded -> there shouldn't be any signals.
    entry_df, bias_df = _entry_df().iloc[:3], _bias_df()
    empty_structure_df = pd.DataFrame()

    signals = DonchianBreakoutStrategy().generate_signals(entry_df, empty_structure_df, bias_df, PARAMS)

    assert signals == []
