"""Backtest mínimo end-to-end (Próximos pasos inmediatos, paso 4 de docs/PLAN.md):
BTC/USDT, timeframe 1h, para validar el pipeline completo (datos -> señal ->
simulación -> reporte) antes de escalar a los 5 mercados x 3 timeframes.

Uso:
    .venv/bin/python scripts/run_minimal_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestConfig, compute_metrics, run_backtest
from src.data.fetcher import earliest_available, fetch_ohlcv
from src.report.report import render_report

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE = "2024-06-01"
UNTIL = "2024-12-01"


def main() -> None:
    print(f"Chequeando fecha de listado de {SYMBOL}...")
    listed_since = earliest_available(SYMBOL, timeframe="1d")
    print(f"  -> primera vela disponible: {listed_since}")

    print(f"Descargando velas {SYMBOL} {TIMEFRAME} de {SINCE} a {UNTIL}...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"  -> {len(df)} velas descargadas")

    if df.empty:
        print("No se descargaron velas — no se puede continuar.")
        return

    cfg = BacktestConfig()
    trades, equity_curve = run_backtest(df, cfg)
    metrics = compute_metrics(trades, equity_curve, cfg.initial_equity)

    print("\nMétricas:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out_path = Path(__file__).resolve().parents[1] / "reports" / "minimal_backtest_report.md"
    render_report(SYMBOL, TIMEFRAME, trades, metrics, out_path)
    print(f"\nReporte escrito en {out_path}")


if __name__ == "__main__":
    main()
