import pandas as pd
import pytest

from trading_lab.analysis.probability import (
    build_probability_table,
    condition_probability,
    forward_direction,
    forward_return,
    one_sample_proportion_z_test,
    signal_hit_probability,
    two_proportion_z_test,
)
from trading_lab.strategy.base import Signal


def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=5 * i)


def test_forward_return_known_values_and_nan_tail():
    df = pd.DataFrame({"close": [100.0, 110.0, 100.0, 90.0, 80.0, 70.0]})
    fwd = forward_return(df, horizon_bars=1)

    assert fwd.iloc[0] == pytest.approx(10.0)
    assert fwd.iloc[1] == pytest.approx((100.0 / 110.0 - 1) * 100.0)
    # The last row doesn't have enough future candles -> NaN.
    assert pd.isna(fwd.iloc[-1])


def test_forward_direction_sign():
    df = pd.DataFrame({"close": [100.0, 110.0, 100.0]})
    direction = forward_direction(df, horizon_bars=1)
    assert direction.iloc[0] == 1.0   # up
    assert direction.iloc[1] == -1.0  # down


def test_condition_probability_known_edge():
    df = pd.DataFrame({"close": [100.0, 110.0, 100.0, 90.0, 80.0, 70.0]})
    # Condition true on the first 2 rows (one up, one down) and false on
    # the next 3 valid rows (all down).
    condition = pd.Series([True, True, False, False, False, False])

    result = condition_probability(df, condition, horizon_bars=1)

    assert result["n"] == 2
    assert result["p_up_condition_pct"] == pytest.approx(50.0)
    # Base = the 5 rows with future data (indices 0-4): 1 of 5 goes up.
    assert result["p_up_base_pct"] == pytest.approx(20.0)
    assert result["edge_pp"] == pytest.approx(30.0)


def test_condition_probability_empty_condition_returns_none_fields():
    df = pd.DataFrame({"close": [100.0, 110.0, 100.0]})
    condition = pd.Series([False, False, False])
    result = condition_probability(df, condition, horizon_bars=1)
    assert result["n"] == 0
    assert result["p_up_condition_pct"] is None
    assert result["edge_pp"] is None


def test_condition_probability_excludes_tail_even_if_condition_true():
    df = pd.DataFrame({"close": [100.0, 110.0]})
    # The only row with a future candle is 0; row 1 (condition true) has
    # no forward_return -> it shouldn't count.
    condition = pd.Series([False, True])
    result = condition_probability(df, condition, horizon_bars=1)
    assert result["n"] == 0


def test_build_probability_table_includes_expected_ingredients():
    n = 5
    merged = pd.DataFrame(
        {
            "timestamp": [_ts(i) for i in range(n)],
            "open": [100.0, 101.0, 99.0, 98.0, 97.0],
            "high": [101.0, 102.0, 100.0, 99.0, 98.0],
            "low": [99.0, 100.0, 98.0, 97.0, 96.0],
            "close": [100.5, 101.5, 98.5, 97.5, 96.5],  # closes near the high -> buying pressure
            "bias_trend": ["up", "up", "up", "down", "down"],
            "struct_trend": ["up", "up", "down", "down", "down"],
            "confirm_bullish": [True, False, False, False, False],
            "confirm_bearish": [False, False, True, False, False],
            "volume_ok": [True, True, True, False, True],
            "session": ["asia", "asia", "london", "london", "new_york"],
        }
    )

    table = build_probability_table(merged, horizon_bars=1)

    assert set(["bullish_bias_1D", "bullish_structure_1h", "bullish_candle_pattern_5m", "sufficient_volume"]).issubset(
        set(table["ingredient"])
    )
    # Individual candle geometry: always computable from the raw OHLC,
    # regardless of what other columns the strategy's DataFrame carries.
    assert set(["bullish_marubozu", "bearish_marubozu", "narrow_range_nr7", "wide_range_wr7", "inside_bar"]).issubset(
        set(table["ingredient"])
    )
    # bullish_bias_1D is True on rows 0,1,2 -> of those, only 0-2 have
    # forward_return (n=4 valid total, horizon=1) -> n counts rows 0,1,2 with future data (0,1,2 have it; row 3 isn't in the condition).
    row = table[table["ingredient"] == "bullish_bias_1D"].iloc[0]
    assert row["n"] == 3  # rows 0,1,2 (all with forward_return available, since horizon=1 and n=5)


def test_two_proportion_z_test_equal_proportions_gives_p_near_one():
    z, p = two_proportion_z_test(50, 100, 50, 100)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_two_proportion_z_test_large_difference_is_significant():
    z, p = two_proportion_z_test(800, 1000, 500, 1000)
    assert abs(z) > 3
    assert p < 0.001


def test_two_proportion_z_test_handles_empty_group():
    z, p = two_proportion_z_test(0, 0, 5, 10)
    assert pd.isna(z) and pd.isna(p)


def test_one_sample_proportion_z_test_matches_baseline_not_significant():
    z, p = one_sample_proportion_z_test(50, 100, 0.5)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_one_sample_proportion_z_test_large_deviation_is_significant():
    z, p = one_sample_proportion_z_test(90, 100, 0.5)
    assert p < 0.001


def test_condition_probability_flags_significant_with_clear_large_sample():
    n = 200
    up_mult, down_mult = 1.03, 0.97
    # The first 99 transitions go up, the rest go down -> perfect separation.
    multipliers = [up_mult if i < 99 else down_mult for i in range(n - 1)]
    closes = [100.0]
    for m in multipliers:
        closes.append(closes[-1] * m)

    df = pd.DataFrame({"close": closes})
    condition = pd.Series([i < 99 for i in range(n)])

    result = condition_probability(df, condition, horizon_bars=1)

    assert result["n"] == 99
    assert result["significant_5pct"] is True
    assert result["p_value"] is not None and result["p_value"] < 0.001


def test_condition_probability_not_significant_with_tiny_sample():
    df = pd.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]})
    condition = pd.Series([True, False, True, False])
    result = condition_probability(df, condition, horizon_bars=1)
    # With so few samples, the observed difference is indistinguishable from chance.
    assert result["significant_5pct"] is False


def test_signal_hit_probability_direction_aware():
    entry_df = pd.DataFrame(
        {
            "timestamp": [_ts(i) for i in range(5)],
            "close": [100.0, 110.0, 90.0, 80.0, 70.0],
        }
    )
    signals = [
        Signal(timestamp=_ts(0), direction="long", entry_price=100.0, stop_loss=95.0, take_profit=110.0,
               confidence_score=4, hour_utc=0, session="asia", reasons={}),
        Signal(timestamp=_ts(1), direction="short", entry_price=110.0, stop_loss=115.0, take_profit=100.0,
               confidence_score=4, hour_utc=0, session="asia", reasons={}),
        Signal(timestamp=_ts(2), direction="long", entry_price=90.0, stop_loss=85.0, take_profit=100.0,
               confidence_score=4, hour_utc=0, session="asia", reasons={}),
    ]

    table = signal_hit_probability(signals, entry_df, horizon_bars=1)

    long_row = table[table["direction"] == "long"].iloc[0]
    short_row = table[table["direction"] == "short"].iloc[0]

    # long at t0 (100->110, +10%, hit) and long at t2 (90->80, -11.11%, miss)
    assert long_row["n"] == 2
    assert long_row["hit_rate_pct"] == pytest.approx(50.0)

    # short at t1 (110->90, -18.18% price move -> +18.18% in favor of the short -> hit)
    assert short_row["n"] == 1
    assert short_row["hit_rate_pct"] == pytest.approx(100.0)
    expected_short_return = -((90.0 / 110.0 - 1) * 100.0)
    assert short_row["avg_return_pct"] == pytest.approx(expected_short_return, abs=0.001)  # rounded to 3 decimals
