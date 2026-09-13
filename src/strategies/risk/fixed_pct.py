"""Risk pieces: a fixed-percentage stop below the setup's reference level,
and a risk/reward-multiple target.

Same math as the engine used before this catalog existed — see
docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

from src.strategies.registry import RISK_STOPS, RISK_TARGETS
from src.strategies.types import EvalContext, SetupResult


@RISK_STOPS.register("fixed_pct")
def fixed_pct_stop(entry_price: float, setup: SetupResult, ctx: EvalContext, params: dict) -> float:
    pct_below_reference = params.get("pct_below_reference", 0.5)
    return setup.reference_level * (1 - pct_below_reference / 100)


@RISK_TARGETS.register("risk_reward")
def risk_reward_target(entry_price: float, stop_price: float, ctx: EvalContext, params: dict) -> float:
    ratio = params.get("ratio", 2.0)
    risk = entry_price - stop_price
    return entry_price + ratio * risk
