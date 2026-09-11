"""Integration test: builds synthetic multi-timeframe entry_df/structure_df/
bias_df, hand-designed to trigger exactly ONE known long signal, and
verifies that `ConfluenceStrategy` detects it with the expected direction,
confidence score, and SL/TP.
"""

import pandas as pd
import pytest

from trading_lab.strategy.confluence_strategy import ConfluenceStrategy, prepare_structure_df

# Same "zigzag" sequence on both higher timeframes: with swing_lookback=1,
# confirms 2 higher swing lows (90->95) and 2 higher swing highs
# (105->115) -> 'up' trend from index 5 onward.
ZIGZAG = [100.0, 90.0, 105.0, 95.0, 115.0, 110.0, 130.0]

PARAMS = {
    "timeframes": {"entry": "5m", "structure": "1h", "bias": "1d"},
    "strategy": {
        "swing_lookback": 1,
        "fib_zone_min": 0.5,
        "fib_zone_max": 0.618,
        "volume_lookback": 3,
        "volume_min_ratio": 0.8,
        "trendline_min_points": 3,
        "trendline_max_points": 6,
        "trendline_proximity_atr_mult": 1.0,
        "trendline_bonus_enabled": True,
        "risk_per_trade_pct": 1.0,
        "atr_period": 3,
        "atr_sl_multiplier": 1.5,
        "reward_risk_ratio": 2.0,
        "commission_pct": 0.1,
        "slippage_pct": 0.05,
        "probability_horizon_bars": 12,
    },
}


def _ohlc_df(values: list[float], timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    """"Flat" candles (high=low=close=open=value) — enough to trigger
    structure/trend detection, which only looks at high/low."""
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": [100.0] * len(values),
        }
    )


def _bias_df() -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=7, freq="1D", tz="UTC")
    return _ohlc_df(ZIGZAG, list(days))


def _structure_df() -> pd.DataFrame:
    hours = pd.date_range("2024-01-07T00:00:00Z", periods=7, freq="1h")
    return _ohlc_df(ZIGZAG, list(hours))


def _entry_df() -> pd.DataFrame:
    # Starts at 05:35 on 2024-01-07 (5 candles of 5m before the signal), so
    # the 'up' bias/structure (confirmed starting at 06:00) is already
    # available right on the signal candle (05:55, closes at 06:00).
    start = pd.Timestamp("2024-01-07T05:35:00Z")
    timestamps = [start + pd.Timedelta(minutes=5 * i) for i in range(7)]

    rows = [
        # open,   high,   low,    close, volume
        (106.0, 107.0, 105.0, 106.0, 100.0),
        (106.0, 106.5, 104.0, 104.5, 100.0),
        (104.5, 105.0, 103.5, 104.0, 100.0),
        (103.5, 104.0, 102.5, 102.8, 100.0),  # bearish, right before the signal
        (102.7, 104.3, 102.5, 104.0, 200.0),  # SIGNAL: bullish engulfing, fib zone (102.64-105), high volume
        (104.0, 104.5, 103.8, 104.2, 100.0),
        (104.2, 104.6, 104.0, 104.3, 100.0),
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = timestamps
    return df


def test_confluence_strategy_detects_single_expected_long_signal():
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()

    signals = ConfluenceStrategy().generate_signals(entry_df, structure_df, bias_df, PARAMS)

    assert len(signals) == 1
    sig = signals[0]

    assert sig.direction == "long"
    assert sig.timestamp == pd.Timestamp("2024-01-07T05:55:00Z")
    assert sig.entry_price == pytest.approx(104.0)
    # Structural stop: swing low of the last structure swing (95).
    assert sig.stop_loss == pytest.approx(95.0)
    # Take profit = entry + reward/risk * risk = 104 + 2*(104-95) = 122.
    assert sig.take_profit == pytest.approx(122.0)
    assert sig.hour_utc == 5
    assert sig.session == "asia"

    # Confidence: 4 mandatory conditions + 1 bonus for engulfing (there
    # aren't enough swings for the geometry bonus, nor a volume ratio > 1.5).
    assert sig.confidence_score == 5
    assert sig.reasons["bias_trend"] == "up"
    assert sig.reasons["structure_trend"] == "up"
    assert sig.reasons["pattern"] == "engulfing"


def test_swing_zone_for_row_skips_non_consecutive_swings_instead_of_crashing():
    from trading_lab.strategy.confluence_strategy import _swing_zone_for_row

    # The last confirmed swing_low (95, idx1) sits at a price HIGHER than
    # the last confirmed swing_high (90, idx0) -- they're not consecutive,
    # they don't form a valid retracement range. Shouldn't crash, should
    # give "no zone".
    out = pd.DataFrame(
        {
            "high": [90.0, 95.0],
            "low": [90.0, 95.0],
            "close": [90.0, 95.0],
            "swing_high": [True, False],
            "swing_low": [False, True],
            "confirmed_at_index": [0, 1],
        }
    )

    zlo, zhi, lo, hi, direction = _swing_zone_for_row(out, i=1, fib_zone_min=0.5, fib_zone_max=0.618)

    assert (zlo, zhi, lo, hi, direction) == (None, None, None, None, None)


def test_swing_zone_for_row_normal_consecutive_swings_still_works():
    from trading_lab.strategy.confluence_strategy import _swing_zone_for_row

    out = pd.DataFrame(
        {
            "high": [90.0, 110.0],
            "low": [90.0, 110.0],
            "close": [90.0, 110.0],
            "swing_high": [False, True],
            "swing_low": [True, False],
            "confirmed_at_index": [0, 1],
        }
    )

    zlo, zhi, lo, hi, direction = _swing_zone_for_row(out, i=1, fib_zone_min=0.5, fib_zone_max=0.618)

    assert lo == pytest.approx(90.0)
    assert hi == pytest.approx(110.0)
    assert direction == "up"
    assert zlo is not None and zhi is not None


def test_trendline_required_filters_out_signal_without_nearby_line():
    # In the known-signal fixture there aren't enough swings (min 3) to fit
    # a support line -> it's never "near" a line. With just the bonus
    # (trendline_required=False, the default behavior) the signal is still
    # generated; if required as a mandatory condition, it should disappear.
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()

    params_required = {
        "timeframes": PARAMS["timeframes"],
        "strategy": {**PARAMS["strategy"], "trendline_required": True},
    }
    signals = ConfluenceStrategy().generate_signals(entry_df, structure_df, bias_df, params_required)

    assert signals == []


def test_confluence_strategy_no_signal_without_trend_alignment():
    # If the daily bias never reaches 'up' (data cut before it's
    # confirmed), no signal should be generated even if every other
    # condition still holds.
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()
    bias_df = bias_df.iloc[:5]  # cut before the bias confirms 'up'

    signals = ConfluenceStrategy().generate_signals(entry_df, structure_df, bias_df, PARAMS)

    assert signals == []
