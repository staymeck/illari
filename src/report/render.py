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


def _stop_noise_lines(trades: pd.DataFrame) -> list[str]:
    """Kaufman (Trading Systems and Methods, ch. 23) calls stops "a duel with
    price noise": a stop can capture the worst exit right before a
    recovery. `stop_was_premature` (see src/backtest/engine.py) measures
    exactly that — among stop-outs, how many would have reached the target
    anyway before the holding window ran out."""
    stopped = trades[trades["exit_reason"] == "stop"]
    if stopped.empty or "stop_was_premature" not in stopped.columns:
        return []

    premature = stopped["stop_was_premature"].fillna(False).astype(bool)
    n_premature = int(premature.sum())
    pct_premature = round(n_premature / len(stopped) * 100, 2)
    return [
        "## Stop-loss noise diagnostic",
        "",
        f"- Stopped-out trades: **{len(stopped)}**",
        f"- Of those, would have reached target anyway (\"premature\" stop): "
        f"**{n_premature} ({pct_premature}%)**",
        "",
        "> A high percentage here means the stop is too tight for this",
        "> market/timeframe's noise, not that the entries were wrong — see",
        "> docs/PLAN.md's Kaufman reference. Consider a wider stop, an",
        "> ATR-based stop (risk piece `atr_multiple`), or `risk.stop.trigger:",
        "> close` to require a close beyond the stop instead of a wick touch.",
        "",
    ]


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
        "> Backtest of a config-driven catalog strategy (see docs/PLAN.md,",
        "> \"Config-driven strategy catalog\", and the strategy's own YAML file",
        "> under config/strategies/ for exactly which pieces it uses). Do not",
        "> treat these numbers as evidence of a profitable strategy on their",
        "> own: check the sample size below, and whether this held up across",
        "> multiple markets/timeframes and out-of-sample — a single run is",
        "> not a conclusion.",
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
        lines += ["## Breakdown by exit reason", ""]
        lines.append(_breakdown_by(trades, "exit_reason").to_markdown(index=False))
        lines += ["", "## Breakdown by entry hour (UTC)", ""]
        lines.append(_breakdown_by(trades, "hour_utc").to_markdown(index=False))
        lines += ["", "## Breakdown by market session", ""]
        lines.append(_breakdown_by(trades, "session").to_markdown(index=False))
        if "pattern" in trades.columns:
            lines += ["", "## Breakdown by confirming candlestick pattern", ""]
            lines.append(_breakdown_by(trades, "pattern").to_markdown(index=False))
        lines.append("")
        lines += _stop_noise_lines(trades)
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
