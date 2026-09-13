"""Renders walk-forward fold results (src.backtest.walk_forward) into a
markdown report — see docs/PLAN.md for why this exists (checking whether a
promising aggregate result holds up across time, not just across markets)."""
from __future__ import annotations

from pathlib import Path

from src.backtest.walk_forward import FoldResult
from src.report.paths import REPORTS_DIR


def render_walk_forward_markdown(
    symbol: str, timeframe: str, strategy_name: str, fold_results: list[FoldResult]
) -> str:
    lines = [
        f"# Walk-forward — {strategy_name}",
        "",
        f"Market: **{symbol}** · Timeframe: **{timeframe}** · Folds: **{len(fold_results)}**",
        "",
        "> Same strategy, unmodified, run independently on each consecutive",
        "> period below. A result driven by one lucky fold — with the others",
        "> flat or negative — is not the same as one that holds up over",
        "> time. See docs/PLAN.md, Kaufman reference.",
        "",
        "| fold | since | until | trades | win_rate_pct | profit_factor | total_return_pct |",
        "|---:|:---|:---|---:|---:|---:|---:|",
    ]
    positive_folds = 0
    for i, r in enumerate(fold_results, start=1):
        m = r.metrics
        if m["total_return_pct"] > 0:
            positive_folds += 1
        lines.append(
            f"| {i} | {r.fold.since} | {r.fold.until} | {m['n_trades']} | "
            f"{m['win_rate_pct']} | {m['profit_factor']} | {m['total_return_pct']} |"
        )

    lines += [
        "",
        f"**{positive_folds}/{len(fold_results)} folds positive.**",
        "",
    ]
    return "\n".join(lines)


def write_walk_forward_report(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    fold_results: list[FoldResult],
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    safe_symbol = symbol.replace("/", "-").lower()
    out_path = reports_dir / "walk_forward" / f"{safe_symbol}_{timeframe}_{strategy_name}.md"
    markdown = render_walk_forward_markdown(symbol, timeframe, strategy_name, fold_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
