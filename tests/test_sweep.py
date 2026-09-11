"""Verifies that configuration variants (overrides) and multi-strategy
support actually change the backtest result — reuses the multi-timeframe
fixture from `test_confluence_strategy.py` (a single known long signal
with the base parameters)."""

import pandas as pd
import pytest

from test_confluence_strategy import PARAMS, _bias_df, _entry_df, _structure_df

from trading_lab.backtest.sweep import (
    apply_overrides,
    iter_strategy_variants,
    rank_variants_by_consistency,
    run_sweep,
    run_variant,
)
from trading_lab.strategy.confluence_strategy import ConfluenceStrategy


def test_apply_overrides_merges_without_mutating_base():
    base = {"a": 1, "b": 2}
    result = apply_overrides(base, {"b": 20, "c": 30})

    assert result == {"a": 1, "b": 20, "c": 30}
    assert base == {"a": 1, "b": 2}  # the original isn't touched


def test_run_variant_reproduces_known_signal_with_base_params():
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()

    signals, result = run_variant(
        ConfluenceStrategy, PARAMS["strategy"], PARAMS["timeframes"], 10000.0, entry_df, structure_df, bias_df
    )

    assert len(signals) == 1
    assert signals[0].direction == "long"
    assert len(result.trades) == 1


def test_run_variant_with_unreachable_fib_zone_produces_no_signal():
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()

    # Zone 0.786-1.0 of the same swing (95-115) falls at 95-99.28 — the
    # known signal's price (104.0) is outside it -> should generate nothing.
    variant_params = apply_overrides(PARAMS["strategy"], {"fib_zone_min": 0.786, "fib_zone_max": 1.0})
    signals, result = run_variant(
        ConfluenceStrategy, variant_params, PARAMS["timeframes"], 10000.0, entry_df, structure_df, bias_df
    )

    assert signals == []
    assert result.trades == []


def _cfg_single_strategy() -> dict:
    return {
        "timeframes": PARAMS["timeframes"],
        "initial_capital": 10000.0,
        "strategies": [
            {
                "name": "confluence",
                "strategy_class": "confluence",
                "base_params": PARAMS["strategy"],
                "variants": [
                    {"name": "base", "overrides": {}},
                    {"name": "impossible_zone", "overrides": {"fib_zone_min": 0.786, "fib_zone_max": 1.0}},
                ],
            }
        ],
    }


def test_iter_strategy_variants_yields_every_combination():
    combos = list(iter_strategy_variants(_cfg_single_strategy()))
    names = [(s, v) for s, _cls, v, _params in combos]
    assert names == [("confluence", "base"), ("confluence", "impossible_zone")]
    assert combos[0][1] is ConfluenceStrategy


def test_run_sweep_covers_every_strategy_variant_and_scenario():
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()
    cfg = _cfg_single_strategy()
    data_by_scenario = {"single_scenario": (entry_df, structure_df, bias_df)}

    sweep_df = run_sweep(cfg, data_by_scenario)

    assert len(sweep_df) == 2  # 2 variants x 1 scenario
    assert set(sweep_df["variant"]) == {"confluence/base", "confluence/impossible_zone"}
    base_row = sweep_df[sweep_df["variant"] == "confluence/base"].iloc[0]
    impossible_row = sweep_df[sweep_df["variant"] == "confluence/impossible_zone"].iloc[0]
    assert base_row["n_trades"] == 1
    assert impossible_row["n_trades"] == 0


def test_run_sweep_covers_multiple_strategies():
    entry_df, structure_df, bias_df = _entry_df(), _structure_df(), _bias_df()
    cfg = _cfg_single_strategy()
    cfg["strategies"].append(
        {
            "name": "breakout_donchian",
            "strategy_class": "breakout",
            "base_params": {
                "donchian_period_bars": 3,
                "swing_lookback": PARAMS["strategy"]["swing_lookback"],
                "volume_lookback": PARAMS["strategy"]["volume_lookback"],
                "atr_period": PARAMS["strategy"]["atr_period"],
                "atr_sl_multiplier": PARAMS["strategy"]["atr_sl_multiplier"],
                "reward_risk_ratio": PARAMS["strategy"]["reward_risk_ratio"],
                "risk_per_trade_pct": PARAMS["strategy"]["risk_per_trade_pct"],
                "commission_pct": PARAMS["strategy"]["commission_pct"],
                "slippage_pct": PARAMS["strategy"]["slippage_pct"],
                "probability_horizon_bars": PARAMS["strategy"]["probability_horizon_bars"],
            },
            "variants": [{"name": "base", "overrides": {}}],
        }
    )
    data_by_scenario = {"single_scenario": (entry_df, structure_df, bias_df)}

    sweep_df = run_sweep(cfg, data_by_scenario)

    variants = set(sweep_df["variant"])
    assert "confluence/base" in variants
    assert "breakout_donchian/base" in variants


def test_rank_variants_by_consistency_penalizes_worst_scenario():
    sweep_df = pd.DataFrame(
        [
            {"variant": "A", "scenario": "s1", "win_rate_pct": 60, "profit_factor": 2.0, "total_return_pct": 10, "n_trades": 5},
            {"variant": "A", "scenario": "s2", "win_rate_pct": 20, "profit_factor": 0.5, "total_return_pct": -5, "n_trades": 5},
            {"variant": "B", "scenario": "s1", "win_rate_pct": 50, "profit_factor": 1.2, "total_return_pct": 3, "n_trades": 5},
            {"variant": "B", "scenario": "s2", "win_rate_pct": 45, "profit_factor": 1.1, "total_return_pct": 2, "n_trades": 5},
        ]
    )
    ranking = rank_variants_by_consistency(sweep_df)

    # B is more consistent across scenarios (worst profit_factor=1.1) than
    # A (worst=0.5), even though A has the best individual result in s1.
    assert ranking.iloc[0]["variant"] == "B"
    assert ranking.iloc[0]["worst_profit_factor"] == pytest.approx(1.1)
