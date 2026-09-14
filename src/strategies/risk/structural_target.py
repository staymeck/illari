"""Risk piece: places the take-profit just below the nearest REAL
resistance above entry (a clustered prior swing high, the current up-leg's
own high, or a Fibonacci retracement level inside it) instead of an
arbitrary risk-multiple (see risk/fixed_pct.py's risk_reward_target).

Mirror image of structural_stop.py, completing the same idea for the other
side of the trade: the entry already uses real structure to decide where
to get in, and structural_stop grounds "how much can I lose" in real
structure below — this grounds "how much can I realistically expect to
gain before something historically real gets in the way" the same way,
instead of just doubling the risk distance and hoping.

A structural ceiling that barely clears the risk being taken isn't a
useful target (that's the whole point of R-multiple targets — reward has
to be worth the risk), so this still falls back to a fixed risk-multiple
target when nothing real is found above entry, or when the nearest real
ceiling doesn't clear `min_risk_reward`.

Diagnosed directly on real data the first time this was tried: the
"nearest resistance above" was almost always a noise-level intermediate
swing high from the retracement's own back-and-forth, not the significant
prior high — this is why it never beat a plain fixed risk-multiple target
(see git history / Bitácora Illari). `min_confluence` (default 1, i.e. no
filtering — preserves that original behavior unless a caller opts in) is
the fix: via src.analysis.structure.nearest_confluent_level, at 2 it skips
a lone, single-source ceiling in favor of the next one out that's also
backed by an independent source (a resistance pivot AND a Fibonacci
level).
"""
from __future__ import annotations

from src.analysis.fibonacci import latest_up_leg, retracement_levels
from src.analysis.structure import nearest_confluent_level, support_resistance_levels
from src.strategies.registry import RISK_TARGETS
from src.strategies.types import EvalContext


@RISK_TARGETS.register("structural_target")
def structural_target(entry_price: float, stop_price: float, ctx: EvalContext, params: dict) -> float:
    order = params.get("order", 3)
    tolerance_pct = params.get("tolerance_pct", 0.5)
    buffer_pct = params.get("buffer_pct", 0.1)
    min_risk_reward = params.get("min_risk_reward", 1.0)
    fallback_ratio = params.get("fallback_ratio", 2.0)
    min_confluence = params.get("min_confluence", 1)

    risk = entry_price - stop_price
    fallback_target = entry_price + fallback_ratio * risk

    resistance_levels = support_resistance_levels(
        ctx.marked_window, order=order, tolerance_pct=tolerance_pct, marked=ctx.marked_window
    )["resistance"]

    fib_levels: list[float] = []
    leg = latest_up_leg(ctx.price_window, order=order, marked=ctx.marked_window)
    if leg is not None:
        fib_levels = list(retracement_levels(leg["low"], leg["high"]).values())
        resistance_levels = resistance_levels + [leg["high"]]  # the leg's own high is a resistance too

    level = nearest_confluent_level(
        entry_price,
        resistance_levels + fib_levels,
        confluence_groups=(resistance_levels, fib_levels),
        direction="above",
        min_confluence=min_confluence,
        tolerance_pct=tolerance_pct,
    )
    if level is None:
        return fallback_target

    # Take profit just before the real ceiling, not exactly at/through it.
    target_price = level * (1 - buffer_pct / 100)

    if risk <= 0 or target_price <= entry_price:
        return fallback_target

    reward = target_price - entry_price
    if reward / risk < min_risk_reward:
        return fallback_target

    return target_price
