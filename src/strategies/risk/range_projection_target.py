"""Risk piece: the classic Wyckoff measured-move target — the
accumulation range's own height, projected up from the breakout ("cause"
projected forward as "effect"). See src.analysis.wyckoff.range_projected_target.

target_fn doesn't receive the setup that fired (unlike stop_fn), so this
re-detects the SAME range wyckoff_accumulation_breakout.py found, using
the same detection function on the same (unchanged) ctx.price_window —
deterministic, so it always recovers the identical range without
trusting a value carried over from the setup. `lookback`/`max_range_pct`
here MUST match the setup piece's own params in the strategy YAML, or
this will detect a different range than the one that actually fired the
trade — documented, not a silent trap.
"""
from __future__ import annotations

from src.analysis.wyckoff import find_trading_range, range_projected_target
from src.strategies.registry import RISK_TARGETS
from src.strategies.types import EvalContext


@RISK_TARGETS.register("range_projection_target")
def range_projection_target(entry_price: float, stop_price: float, ctx: EvalContext, params: dict) -> float:
    lookback = params.get("lookback", 20)
    max_range_pct = params.get("max_range_pct", 8.0)
    fallback_ratio = params.get("fallback_ratio", 2.0)
    # A 1:1 Wyckoff measured-move target and a stop just past the OTHER
    # side of the same range are geometrically close to the same
    # distance (both roughly the range's own height) - a min_risk_reward
    # near 1.0 would reject almost every real breakout this piece is
    # meant to catch. 0.5 is a basic sanity floor, not a demand for
    # reward > risk (that edge is meant to come from win rate/timing,
    # not from this piece's own reward skew).
    min_risk_reward = params.get("min_risk_reward", 0.5)

    risk = entry_price - stop_price
    fallback_target = entry_price + fallback_ratio * risk

    range_info = find_trading_range(ctx.price_window, lookback=lookback, max_range_pct=max_range_pct)
    if range_info is None:
        return fallback_target

    target = range_projected_target(entry_price, range_info["high"], range_info["low"], direction="long")

    if risk <= 0 or target <= entry_price:
        return fallback_target

    reward = target - entry_price
    if reward / risk < min_risk_reward:
        return fallback_target

    return target
