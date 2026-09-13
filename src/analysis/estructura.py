"""Estructura de mercado: máximos/mínimos, tendencia y niveles de
soporte/resistencia — calculados con reglas objetivas de código, no "a ojo".

Referencia: docs/PLAN.md — "Componentes del motor de análisis técnico", punto 1;
y resources/12_claves_analisis_tecnico.pdf (Teoría de Dow, soporte y resistencia).
"""
from __future__ import annotations

import pandas as pd

# Un swing point (pivote) requiere `order` velas más bajas/altas a cada lado
# para confirmarse — evita marcar ruido de una sola vela como pivote real.
_DEFAULT_ORDER = 3


def find_swing_points(df: pd.DataFrame, order: int = _DEFAULT_ORDER) -> pd.DataFrame:
    """Marca pivotes de máximo y mínimo confirmados (fractal simple: el pivote
    es el extremo dentro de una ventana de `2*order + 1` velas centrada en él).

    Devuelve el mismo DataFrame con dos columnas booleanas nuevas:
    `is_swing_high`, `is_swing_low`.
    """
    highs = df["high"]
    lows = df["low"]
    window = 2 * order + 1

    is_swing_high = highs.rolling(window, center=True).apply(
        lambda w: w.iloc[order] == w.max(), raw=False
    ).fillna(0).astype(bool)
    is_swing_low = lows.rolling(window, center=True).apply(
        lambda w: w.iloc[order] == w.min(), raw=False
    ).fillna(0).astype(bool)

    out = df.copy()
    out["is_swing_high"] = is_swing_high
    out["is_swing_low"] = is_swing_low
    return out


def classify_trend(df: pd.DataFrame, order: int = _DEFAULT_ORDER, lookback_swings: int = 2) -> str:
    """Clasifica la tendencia según la Teoría de Dow: compara los últimos
    `lookback_swings` máximos y mínimos confirmados.

    - "alcista": máximos y mínimos crecientes
    - "bajista": máximos y mínimos decrecientes
    - "lateral": no hay swings suficientes o no son consistentes
    """
    marked = find_swing_points(df, order=order)
    highs = marked.loc[marked["is_swing_high"], "high"].tail(lookback_swings)
    lows = marked.loc[marked["is_swing_low"], "low"].tail(lookback_swings)

    if len(highs) < lookback_swings or len(lows) < lookback_swings:
        return "lateral"

    highs_rising = highs.is_monotonic_increasing
    lows_rising = lows.is_monotonic_increasing
    highs_falling = highs.is_monotonic_decreasing
    lows_falling = lows.is_monotonic_decreasing

    if highs_rising and lows_rising:
        return "alcista"
    if highs_falling and lows_falling:
        return "bajista"
    return "lateral"


def support_resistance_levels(
    df: pd.DataFrame, order: int = _DEFAULT_ORDER, tolerance_pct: float = 0.5
) -> dict[str, list[float]]:
    """Agrupa los swing lows en niveles de soporte y los swing highs en niveles
    de resistencia, fusionando pivotes que caen dentro de `tolerance_pct`% entre
    sí (para no reportar 10 "niveles" que en la práctica son el mismo)."""
    marked = find_swing_points(df, order=order)

    def _cluster(values: pd.Series) -> list[float]:
        levels: list[float] = []
        for value in sorted(values.tolist()):
            if levels and abs(value - levels[-1]) / levels[-1] * 100 <= tolerance_pct:
                levels[-1] = (levels[-1] + value) / 2  # funde con el nivel cercano
            else:
                levels.append(value)
        return levels

    supports = _cluster(marked.loc[marked["is_swing_low"], "low"])
    resistances = _cluster(marked.loc[marked["is_swing_high"], "high"])
    return {"support": supports, "resistance": resistances}
