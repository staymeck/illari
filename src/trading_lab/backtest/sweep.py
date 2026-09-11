"""Runs different STRATEGIES (confluence, breakout, ...) and, within each
one, different CONFIGURATIONS (variants), over the same already-downloaded
scenarios — to compare which performs better and, above all, which
performs CONSISTENTLY better across both scenarios (not overfit to just
one).
"""

from __future__ import annotations

import pandas as pd

from trading_lab.backtest import metrics
from trading_lab.backtest.engine import run_backtest
from trading_lab.strategy.base import Strategy
from trading_lab.strategy.registry import get_strategy_class

ScenarioData = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # entry_df, structure_df, bias_df


def apply_overrides(base_params: dict, overrides: dict) -> dict:
    merged = dict(base_params)
    merged.update(overrides)
    return merged


def run_variant(
    strategy_cls: type[Strategy],
    variant_params: dict,
    timeframes: dict,
    initial_capital: float,
    entry_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    bias_df: pd.DataFrame,
):
    """Runs signal generation + backtest for one concrete strategy, with an
    already-resolved parameter variant, over an already-downloaded
    scenario (doesn't hit the API again)."""
    variant_cfg = {"timeframes": timeframes, "strategy": variant_params}
    strategy = strategy_cls()
    signals = strategy.generate_signals(entry_df, structure_df, bias_df, variant_cfg)
    result = run_backtest(
        signals,
        entry_df,
        initial_capital=initial_capital,
        risk_per_trade_pct=variant_params["risk_per_trade_pct"],
        commission_pct=variant_params["commission_pct"],
        slippage_pct=variant_params["slippage_pct"],
    )
    return signals, result


def iter_strategy_variants(cfg: dict):
    """Iterates (strategy_name, strategy_cls, variant_name, variant_params)
    for every strategy × variant defined in `cfg["strategies"]`."""
    for strategy_cfg in cfg.get("strategies", []):
        strategy_cls = get_strategy_class(strategy_cfg["strategy_class"])
        base_params = strategy_cfg.get("base_params", {})
        variants = strategy_cfg.get("variants") or [{"name": "base", "overrides": {}}]
        for variant in variants:
            variant_params = apply_overrides(base_params, variant.get("overrides", {}))
            yield strategy_cfg["name"], strategy_cls, variant["name"], variant_params


def run_sweep(cfg: dict, data_by_scenario: dict[str, ScenarioData]) -> pd.DataFrame:
    """Runs every strategy × variant from `cfg["strategies"]` over each
    scenario in `data_by_scenario`. Returns a "<strategy>/<variant>" ×
    scenario table with the metrics for each combination."""
    rows = []
    for strategy_name, strategy_cls, variant_name, variant_params in iter_strategy_variants(cfg):
        row_name = f"{strategy_name}/{variant_name}"
        for scenario_name, (entry_df, structure_df, bias_df) in data_by_scenario.items():
            _, result = run_variant(
                strategy_cls, variant_params, cfg["timeframes"], cfg["initial_capital"], entry_df, structure_df, bias_df
            )
            summary = metrics.summarize(result, cfg["initial_capital"])
            rows.append({"variant": row_name, "scenario": scenario_name, **summary})

    return pd.DataFrame(rows)


def rank_variants_by_consistency(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """For each "<strategy>/<variant>" row, takes the WORST result across
    scenarios (min of profit_factor and total_return) — a combination that
    only performs well in one scenario and poorly in the other gets
    penalized, instead of rewarding configurations overfit to a single
    period."""
    if sweep_df.empty:
        return sweep_df

    grouped = sweep_df.groupby("variant").agg(
        n_scenarios=("scenario", "count"),
        worst_win_rate_pct=("win_rate_pct", "min"),
        worst_profit_factor=("profit_factor", "min"),
        worst_return_pct=("total_return_pct", "min"),
        total_trades=("n_trades", "sum"),
    )
    return grouped.sort_values("worst_profit_factor", ascending=False).reset_index()
