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
    title: str = "Backtest report — Trading lab",
) -> Path:
    """`chart_paths[display_name][scenario]` = {"candles": Path, "equity": Path},
    where `display_name` is "<strategy>/<variant>" for each combination
    marked `detailed: true`. `probability_tables`/`signal_hit_tables`
    (optional): [display_name][scenario] -> output of `analysis/probability.py`.
    `sweep_df` (optional): output of `backtest/sweep.py::run_sweep`, already
    with every strategy × variant × scenario (row names
    "<strategy>/<variant>"). `title` (optional): lets a per-market report
    (see `build_cross_market_summary`) identify itself in its own heading.
    """
    probability_tables = probability_tables or {}
    signal_hit_tables = signal_hit_tables or {}

    lines: list[str] = [f"# {title}\n"]

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


def build_cross_market_summary(
    all_sweep_df: pd.DataFrame,
    output_path: Path,
    market_report_paths: dict[str, Path],
) -> Path:
    """Top-level report, one level above each market's own detailed report
    (`build_report`, which keeps its full per-strategy detail — charts,
    hour/session breakdown, probability tables — scoped to a single
    market). `all_sweep_df`: every market × strategy/variant × scenario row
    (output of `cli.py::_run_market_backtest`, `sweep_df` with a `market`
    column prepended). `market_report_paths`: market name -> its own
    report.md path, to link out to the full detail.
    """
    lines: list[str] = ["# Trading lab — cross-market summary\n"]

    if all_sweep_df is None or all_sweep_df.empty:
        lines.append("_No data._\n")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    markets = sorted(all_sweep_df["market"].unique())
    lines.append(
        f"Every strategy/variant combination run against **{len(markets)} markets** "
        f"({', '.join(markets)}) × both scenarios — arbitrariness on purpose, so a "
        "result isn't just one asset's quirk wearing different names. Each market's "
        "own report keeps the full detail (charts, hour/session breakdown, "
        "per-ingredient probability tables); this page only compares across all of "
        "them.\n"
    )

    lines.append("## Per-market reports\n")
    for market_name in markets:
        path = market_report_paths.get(market_name)
        if path is not None:
            lines.append(f"- [{market_name}]({path.relative_to(output_path.parent).as_posix()})")
    lines.append("")

    lines.append("## Full comparison (every market × variant × scenario)\n")
    pivot_cols = ["market", "variant", "scenario", "n_trades", "win_rate_pct", "profit_factor", "total_return_pct", "max_drawdown_pct"]
    lines.append(_df_to_markdown_table(all_sweep_df[pivot_cols]) + "\n")

    lines.append("## Cross-market consistency ranking (worst case across ALL markets × scenarios)\n")
    lines.append(
        "The strictest version of the consistency ranking: for each strategy/variant, "
        "this takes the single WORST result among every market × scenario combination "
        "it ran on — not just the worst of 2 scenarios on 1 market, but the worst of "
        "up to 10 combinations. A configuration that looks good here had to hold up "
        "across genuinely different assets, not just different time periods of the "
        "same one.\n"
    )
    ranking = sweep_module.rank_variants_by_consistency(all_sweep_df)
    lines.append(_df_to_markdown_table(ranking) + "\n")

    lines.append("## Per-market consistency ranking\n")
    lines.append(
        "Same ranking, but computed separately within each market (worst case across "
        "its own 2 scenarios only) — useful to see whether a variant's cross-market "
        "ranking above is being dragged down by one specific asset rather than failing "
        "everywhere.\n"
    )
    for market_name in markets:
        lines.append(f"### {market_name}\n")
        market_ranking = sweep_module.rank_variants_by_consistency(all_sweep_df[all_sweep_df["market"] == market_name])
        lines.append(_df_to_markdown_table(market_ranking) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
