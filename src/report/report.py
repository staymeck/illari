"""Genera un report.md con métricas globales y desglose por hora/sesión,
mismo formato que resources/report.md (el reporte de la sesión anterior)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _breakdown_by(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = trades.groupby(column)
    rows = []
    for key, group in grouped:
        wins = group[group["pnl_abs"] > 0]
        gross_win = wins["pnl_abs"].sum()
        gross_loss = -group[group["pnl_abs"] <= 0]["pnl_abs"].sum()
        rows.append(
            {
                column: key,
                "n_trades": len(group),
                "win_rate_pct": round(len(wins) / len(group) * 100, 2),
                "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
                "avg_pnl": round(group["pnl_abs"].mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(column)


def render_report(
    symbol: str,
    timeframe: str,
    trades: pd.DataFrame,
    metrics: dict,
    out_path: Path,
) -> None:
    lines = [
        "# Reporte de backtest — Laboratorio de trading (validación mínima del pipeline)",
        "",
        f"Mercado: **{symbol}** · Timeframe: **{timeframe}**",
        "",
        "> Corrida de validación del motor end-to-end (solo estructura de mercado,",
        "> sin Fibonacci/velas/volumen todavía — ver docs/PLAN.md). No usar estos",
        "> números como evidencia de una estrategia rentable: falta walk-forward,",
        "> out-of-sample, y una muestra de operaciones bastante más grande.",
        "",
        "## Métricas globales",
        "",
        f"- Operaciones: **{metrics['n_trades']}**",
        f"- Win rate: **{metrics['win_rate_pct']}%**",
        f"- Profit factor: **{metrics['profit_factor']}**",
        f"- Expectancy: **{metrics['expectancy']}**",
        f"- Drawdown máximo: **{metrics['max_drawdown_pct']}%**",
        f"- Retorno total: **{metrics['total_return_pct']}%**",
        f"- Equity final: **{metrics['final_equity']}**",
        "",
    ]

    if not trades.empty:
        lines += ["## Desglose por hora de entrada (UTC)", ""]
        lines.append(_breakdown_by(trades, "hour_utc").to_markdown(index=False))
        lines += ["", "## Desglose por sesión de mercado", ""]
        lines.append(_breakdown_by(trades, "session").to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No se generó ninguna operación en el rango simulado._")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
