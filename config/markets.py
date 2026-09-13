"""Configuración de mercados, timeframes y sesiones horarias del laboratorio.

Referencia: docs/PLAN.md — sección "Alcance de la Fase 1".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    symbol: str  # símbolo spot en Binance, formato ccxt (ej. "BTC/USDT")
    role: str    # por qué está en el set (documentación, no usado en cálculo)


# Los 5 mercados acordados en docs/PLAN.md. SOL/USDT elegido sobre BNB/USDT como
# altcoin de mayor beta; se puede correr scripts/check_listing_dates.py para
# confirmar la fecha real de listado de cada uno antes de fijar escenarios.
MARKETS: list[Market] = [
    Market("BTC/USDT", "referencia base, cripto blue chip"),
    Market("ETH/USDT", "blue chip con dinámica propia (DeFi/contratos)"),
    Market("PAXG/USDT", "oro tokenizado — único activo no-cripto real del set"),
    Market("SOL/USDT", "altcoin de mayor beta/volatilidad"),
    Market("DOGE/USDT", "guiada por sentimiento/redes, no por fundamentos"),
]

# Timeframes multi-contexto: fino (entradas), intermedio y global.
TIMEFRAMES: list[str] = ["5m", "1h", "1d"]

# Sesiones de mercado en UTC (horas de inicio/fin, fin exclusivo). Se usan para
# etiquetar cada operación simulada y desglosar resultados por sesión.
SESSIONS_UTC: dict[str, tuple[int, int]] = {
    "asia": (0, 8),
    "london": (7, 9),              # apertura de Londres, se resuelve solapamiento abajo
    "overlap_london_ny": (12, 16),
    "new_york": (13, 21),
    "off_hours": (21, 24),
}


def session_for_hour(hour_utc: int) -> str:
    """Devuelve la sesión de mercado (UTC) para una hora dada, con prioridad
    al solapamiento Londres-NY cuando corresponde, igual que en el reporte previo
    (resources/report.md)."""
    if 12 <= hour_utc < 16:
        return "overlap_london_ny"
    if 13 <= hour_utc < 21:
        return "new_york"
    if 7 <= hour_utc < 12:
        return "london"
    if 0 <= hour_utc < 8:
        return "asia"
    return "off_hours"
