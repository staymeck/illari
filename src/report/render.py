"""Renders a single backtest run into a markdown report.

Rendering (`render_run_markdown`) is a pure function — no filesystem access —
so it's testable without touching disk; `write_run_report` is the thin
wrapper that saves it under ./reports (see src/report/paths.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report.paths import REPORTS_DIR, run_report_path


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


def render_run_markdown(
    symbol: str,
    timeframe: str,
    trades: pd.DataFrame,
    metrics: dict,
    scenario: str = "default",
) -> str:
    """Builds the markdown text for one backtest run. Same metric format as
    resources/report.md (the report from the previous session)."""
    lines = [
        "# Backtest report — Trading lab",
        "",
        f"Market: **{symbol}** · Timeframe: **{timeframe}** · Scenario: **{scenario}**",
        "",
        "> End-to-end validation run of the engine (structure + Fibonacci",
        "> confluence + candlestick + volume confirmation; funding rate/Fear &",
        "> Greed not integrated yet — see docs/PLAN.md). Do not treat these",
        "> numbers as evidence of a profitable strategy: with every confluence",
        "> filter stacked, the sample size on a single market/timeframe is now",
        "> too small to conclude anything — this is expected at this stage,",
        "> and exactly why the plan calls for walk-forward validation,",
        "> out-of-sample testing, and scaling to 5 markets x 3 timeframes",
        "> before drawing any real conclusion.",
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
        if "pattern" in trades.columns:
            lines += ["", "## Breakdown by confirming candlestick pattern", ""]
            lines.append(_breakdown_by(trades, "pattern").to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No trades were generated in the simulated range._")

    return "\n".join(lines)


def write_run_report(
    symbol: str,
    timeframe: str,
    trades: pd.DataFrame,
    metrics: dict,
    scenario: str = "default",
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Renders and saves a single run's report under `reports_dir`, following
    the standard naming from src/report/paths.py. Returns the path written."""
    out_path = run_report_path(symbol, timeframe, scenario, reports_dir=reports_dir)
    markdown = render_run_markdown(symbol, timeframe, trades, metrics, scenario=scenario)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
