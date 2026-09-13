"""Generates a report.md with overall metrics and an hour/session breakdown,
same format as resources/report.md (the report from the previous session)."""
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
        "# Backtest report — Trading lab (minimal pipeline validation)",
        "",
        f"Market: **{symbol}** · Timeframe: **{timeframe}**",
        "",
        "> End-to-end validation run of the engine (market structure only,",
        "> no Fibonacci/candlesticks/volume yet — see docs/PLAN.md). Do not",
        "> treat these numbers as evidence of a profitable strategy: this is",
        "> missing walk-forward validation, out-of-sample testing, and a much",
        "> larger trade sample.",
        "",
        "## Overall metrics",
        "",
        f"- Trades: **{metrics['n_trades']}**",
        f"- Win rate: **{metrics['win_rate_pct']}%**",
        f"- Profit factor: **{metrics['profit_factor']}**",
        f"- Expectancy: **{metrics['expectancy']}**",
        f"- Max drawdown: **{metrics['max_drawdown_pct']}%**",
        f"- Total return: **{metrics['total_return_pct']}%**",
        f"- Final equity: **{metrics['final_equity']}**",
        "",
    ]

    if not trades.empty:
        lines += ["## Breakdown by entry hour (UTC)", ""]
        lines.append(_breakdown_by(trades, "hour_utc").to_markdown(index=False))
        lines += ["", "## Breakdown by market session", ""]
        lines.append(_breakdown_by(trades, "session").to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No trades were generated in the simulated range._")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
