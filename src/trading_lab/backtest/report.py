"""Generates the markdown report: configuration comparison (all
strategies × variants × scenarios), and the detail of each strategy/variant
combination marked `detailed: true` in `settings.yaml` (metrics, hour/
session breakdown, per-ingredient probability with significance test, and
links to the HTML charts).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_lab.backtest import metrics, sweep as sweep_module
from trading_lab.backtest.engine import BacktestResult

# display_name ("<strategy>/<variant>") -> scenario_name -> value
NestedResults = dict[str, dict[str, BacktestResult]]
NestedTables = dict[str, dict[str, pd.DataFrame]]
NestedCharts = dict[str, dict[str, dict[str, Path]]]


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._\n"
    return df.to_markdown(index=False)


def build_report(
    strategies_scenario_results: NestedResults,
    initial_capital: float,
    chart_paths: NestedCharts,
    output_path: Path,
    probability_tables: NestedTables | None = None,
    signal_hit_tables: NestedTables | None = None,
    sweep_df: pd.DataFrame | None = None,
) -> Path:
    """`chart_paths[display_name][scenario]` = {"candles": Path, "equity": Path},
    where `display_name` is "<strategy>/<variant>" for each combination
    marked `detailed: true`. `probability_tables`/`signal_hit_tables`
    (optional): [display_name][scenario] -> output of `analysis/probability.py`.
    `sweep_df` (optional): output of `backtest/sweep.py::run_sweep`, already
    with every strategy × variant × scenario (row names
    "<strategy>/<variant>").
    """
    probability_tables = probability_tables or {}
    signal_hit_tables = signal_hit_tables or {}

    lines: list[str] = ["# Backtest report — Trading lab\n"]

    if sweep_df is not None and not sweep_df.empty:
        lines.append("## Configuration comparison (all strategies)\n")
        lines.append(
            "Every strategy/variant combination run against both scenarios. "
            "The ranking table summarizes, per combination, the **worst** "
            "result across the two scenarios (not the average) — so we don't "
            "pick something that only works in one period, and to penalize "
            "overfitting.\n"
        )
        pivot_cols = ["variant", "scenario", "n_trades", "win_rate_pct", "profit_factor", "total_return_pct", "max_drawdown_pct"]
        lines.append(_df_to_markdown_table(sweep_df[pivot_cols]) + "\n")

        lines.append("### Consistency ranking (worst case across scenarios)\n")
        ranking = sweep_module.rank_variants_by_consistency(sweep_df)
        lines.append(_df_to_markdown_table(ranking) + "\n")

    for display_name, scenario_results in strategies_scenario_results.items():
        lines.append(f"## {display_name}\n")

        summary_rows = []
        for scenario_name, result in scenario_results.items():
            s = metrics.summarize(result, initial_capital)
            s["scenario"] = scenario_name
            summary_rows.append(s)
        summary_df = pd.DataFrame(summary_rows)[
            ["scenario", "n_trades", "win_rate_pct", "profit_factor", "expectancy",
             "max_drawdown_pct", "total_return_pct", "final_equity"]
        ]
        lines.append(_df_to_markdown_table(summary_df) + "\n")

        for scenario_name, result in scenario_results.items():
            lines.append(f"### Scenario: {scenario_name}\n")

            s = metrics.summarize(result, initial_capital)
            lines.append(
                f"- Trades: **{s['n_trades']}** · Win rate: **{s['win_rate_pct']}%** · "
                f"Profit factor: **{s['profit_factor']}** · Total return: **{s['total_return_pct']}%** · "
                f"Max drawdown: **{s['max_drawdown_pct']}%**\n"
            )

            paths = chart_paths.get(display_name, {}).get(scenario_name, {})
            if "candles" in paths:
                lines.append(f"[View interactive candlestick chart]({paths['candles'].relative_to(output_path.parent).as_posix()})\n")
            if "equity" in paths:
                lines.append(f"[View interactive equity curve]({paths['equity'].relative_to(output_path.parent).as_posix()})\n")

            lines.append("#### Breakdown by entry hour (UTC)\n")
            lines.append(_df_to_markdown_table(metrics.breakdown_by(result, "hour_utc")) + "\n")

            lines.append("#### Breakdown by market session\n")
            lines.append(_df_to_markdown_table(metrics.breakdown_by(result, "session")) + "\n")

            prob_table = probability_tables.get(display_name, {}).get(scenario_name)
            if prob_table is not None:
                lines.append(
                    "#### Per-ingredient probability analysis (1h horizon)\n\n"
                    "For each condition on its own: probability that price rises "
                    "in the next hour *given* the condition held, versus its "
                    "complement (condition false) — the \"edge\" is the difference "
                    "in percentage points against the overall base rate, and "
                    "`p_value`/`significant_5pct` come from a two-sample "
                    "proportion test (condition vs. complement): a large edge with "
                    "a small `n` may not be significant — don't mistake luck for "
                    "a real edge. This is pure after-the-fact analytics (it looks "
                    "ahead on purpose, to measure), never used to generate the "
                    "signal in real time.\n"
                )
                lines.append(_df_to_markdown_table(prob_table) + "\n")

            hit_table = signal_hit_tables.get(display_name, {}).get(scenario_name)
            if hit_table is not None:
                lines.append("#### Full-signal hit probability\n\n")
                lines.append(
                    "For the signals actually generated: probability that price "
                    "moved in favor of the signal's direction over the following "
                    "hour, versus `p_base_pct` (the market's base probability of "
                    "moving in that direction, not 50%) — with its own "
                    "significance test.\n"
                )
                lines.append(_df_to_markdown_table(hit_table) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
