"""Tests on synthetic data for src/analysis/volume.py."""
import pandas as pd

from src.analysis.volume import (
    confirms_buyer_pressure,
    directional_volume_bias,
    is_volume_spike,
    relative_volume,
)


def _candles(rows: list[dict]) -> pd.DataFrame:
    """Builds candles from {open, close, volume} dicts; high/low are derived
    from the body since these tests don't care about wicks."""
    out = []
    for r in rows:
        out.append(
            {
                "open": r["open"],
                "close": r["close"],
                "high": max(r["open"], r["close"]),
                "low": min(r["open"], r["close"]),
                "volume": r["volume"],
            }
        )
    return pd.DataFrame(out)


def test_relative_volume_known_ratio():
    # Steady volume of 10, then a spike to 50 on the last candle.
    df = _candles([{"open": 1, "close": 1, "volume": 10}] * 4 + [{"open": 1, "close": 1, "volume": 50}])

    rel = relative_volume(df, window=3)

    assert rel.iloc[-1] == 5.0  # 50 / mean(10, 10, 10)


def test_is_volume_spike_true_and_false():
    df = _candles([{"open": 1, "close": 1, "volume": 10}] * 4 + [{"open": 1, "close": 1, "volume": 50}])

    assert is_volume_spike(df, idx=4, window=3, threshold=1.5)
    assert not is_volume_spike(df, idx=3, window=3, threshold=1.5)


def test_directional_volume_bias_known_value():
    df = _candles(
        [
            {"open": 1, "close": 2, "volume": 10},  # bullish
            {"open": 2, "close": 1, "volume": 5},  # bearish
            {"open": 1, "close": 3, "volume": 15},  # bullish
            {"open": 3, "close": 1, "volume": 10},  # bearish
        ]
    )

    bias = directional_volume_bias(df, window=4)

    assert bias == 0.25  # (25 - 15) / 40


def test_directional_volume_bias_empty_window_is_zero():
    df = _candles([{"open": 1, "close": 2, "volume": 10}]).iloc[:0]  # same columns, no rows

    assert directional_volume_bias(df, window=5) == 0.0


def test_confirms_buyer_pressure_true_when_spike_and_buyer_dominant():
    rows = [{"open": 1, "close": 2, "volume": 10}] * 3 + [{"open": 2, "close": 1, "volume": 5}]
    rows += [{"open": 1, "close": 2, "volume": 60}]  # spike, and still net buyer-dominant
    df = _candles(rows)

    assert confirms_buyer_pressure(df, idx=4, window=4, spike_threshold=1.2, min_bias=0.1)


def test_confirms_buyer_pressure_false_when_seller_dominant():
    rows = [{"open": 2, "close": 1, "volume": 10}] * 4  # all bearish
    rows += [{"open": 2, "close": 1, "volume": 60}]  # spike, but seller-dominant
    df = _candles(rows)

    assert not confirms_buyer_pressure(df, idx=4, window=4, spike_threshold=1.2, min_bias=0.1)


def test_confirms_buyer_pressure_false_without_spike():
    df = _candles([{"open": 1, "close": 2, "volume": 10}] * 5)  # flat volume, no spike

    assert not confirms_buyer_pressure(df, idx=4, window=3, spike_threshold=1.2, min_bias=0.1)
