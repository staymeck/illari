"""Cross-run comparison report: one row per (symbol, timeframe, scenario) run,
plus links to each individual report — the view that becomes necessary once
the lab scales past a single run (5 markets x 3 timeframes, see docs/PLAN.md).

Mirrors the "Resumen comparativo" table at the top of the old
resources/report.md, generalized to any number of runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.report.paths import REPORTS_DIR, run_report_path, summary_report_path


@dataclass(frozen=True)
class RunResult:
    """One backtest run's identity + metrics, as needed for the summary row
    and its link to the full individual report."""

    symbol: str
    timeframe: str
    metrics: dict
    scenario: str = "default"


def render_summary_markdown(results: list[RunResult]) -> str:
    if not results:
        return "# Backtest lab — summary across runs\n\n_No runs yet._\n"

    rows = [
        {"symbol": r.symbol, "timeframe": r.timeframe, "scenario": r.scenario, **r.metrics}
        for r in results
    ]
    table = pd.DataFrame(rows)

    lines = [
        "# Backtest lab — summary across runs",
        "",
        table.to_markdown(index=False),
        "",
        "## Individual reports",
        "",
    ]
    for r in results:
        path = run_report_path(r.symbol, r.timeframe, r.scenario)
        lines.append(f"- [{r.symbol} · {r.timeframe} · {r.scenario}]({path.name})")
    lines.append("")
    return "\n".join(lines)


def write_summary_report(results: list[RunResult], reports_dir: Path = REPORTS_DIR) -> Path:
    out_path = summary_report_path(reports_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_summary_markdown(results), encoding="utf-8")
    return out_path
