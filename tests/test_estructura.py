"""Tests sobre datos sintéticos para src/analysis/estructura.py — casos donde
se conoce de antemano cuál debe ser el resultado (ver docs/PLAN.md, Verificación)."""
import pandas as pd

from src.analysis.estructura import (
    classify_trend,
    find_swing_points,
    support_resistance_levels,
)


def _candles(closes: list[float]) -> pd.DataFrame:
    """Construye velas sintéticas simples: high = low = open = close, con un
    timestamp horario incremental — suficiente para testear geometría pura."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def test_find_swing_points_marks_known_peak_and_valley():
    # Serie con un pico claro en el medio y un valle claro después.
    closes = [1, 2, 3, 10, 3, 2, 1, 0.5, 1, 2, 3]
    df = _candles(closes)

    marked = find_swing_points(df, order=2)

    assert marked.loc[3, "is_swing_high"]  # el valor 10
    assert marked.loc[7, "is_swing_low"]  # el valor 0.5


def test_classify_trend_rising_highs_and_lows_is_alcista():
    # Dos "olas" con máximos y mínimos claramente crecientes.
    closes = [1, 2, 1.5, 3, 2.5, 5, 4, 7]
    df = _candles(closes)

    assert classify_trend(df, order=1, lookback_swings=2) == "alcista"


def test_classify_trend_falling_highs_and_lows_is_bajista():
    closes = [7, 4, 5, 2.5, 3, 1.5, 2, 1]
    df = _candles(closes)

    assert classify_trend(df, order=1, lookback_swings=2) == "bajista"


def test_classify_trend_without_enough_swings_is_lateral():
    closes = [1, 2, 3, 4, 5]  # tendencia monótona, sin pivotes intermedios
    df = _candles(closes)

    assert classify_trend(df, order=2, lookback_swings=2) == "lateral"


def test_support_resistance_merges_nearby_levels():
    # Dos mínimos casi idénticos (dentro de tolerancia) deben fundirse en 1 solo.
    closes = [5, 3, 5, 3.01, 5, 8, 5]
    df = _candles(closes)

    levels = support_resistance_levels(df, order=1, tolerance_pct=1.0)

    assert len(levels["support"]) == 1
    assert 3.0 <= levels["support"][0] <= 3.01
