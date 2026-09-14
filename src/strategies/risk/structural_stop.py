"""Risk piece: places the stop at the nearest REAL structural level below
entry (a clustered prior support, or the next Fibonacci retracement level
down) instead of an arbitrary fixed percentage (see risk/fixed_pct.py).

Motivated directly by an inconsistency the user pointed out: the entry
already uses real structure (support_touch, fibonacci confluence) to decide
WHERE to get in, but fixed_pct_stop then throws all of that away and uses
an arbitrary % to decide how much risk to take. This doesn't change WHETHER
a trade is taken — that's still entirely up to context/setup/confirmations
— only HOW its stop is measured, grounding "how much can I lose" in the
same geometry the entry itself relied on ("punto de equilibrio": if price
breaks this level, there's no known structure catching it here).

`min_confluence` (default 1, i.e. no filtering — preserves the original
behavior unless a caller opts in): how many independent structural signals
(a support cluster AND a Fibonacci level) must agree near a candidate level
before it's trusted as the stop, via
src.analysis.structure.nearest_confluent_level. At 1, the nearest level is
used no matter its source, same as before; at 2, a lone, noise-level pivot
with nothing else backing it is skipped in favor of the next real level
further out — a rough proxy for "liquidity" (real prior interest) versus a
line price could cut through easily.

`buffer_pct` vs. `max_stop_pct`, a real interaction to know about: since
support_touch already requires entry within ~1% of the SAME level this
piece then anchors to, a small buffer_pct produces a very tight stop (more
exposure to ordinary noise, not a wider "safer" one). Conversely, as
buffer_pct approaches max_stop_pct, every real level's stop distance starts
exceeding the ceiling and the fallback branch fires on nearly every trade —
silently degenerating into a fixed-percentage stop and defeating the whole
point of this piece. A swept comparison across all 5 markets found
buffer_pct=2.0 (with the default max_stop_pct=5.0) the best point in range
0.1-7.0 — see config/strategies/trend_pullback_htf_structural_stop.yaml's
header comment for the full numbers and the overfitting caveat.
"""
from __future__ import annotations

from src.analysis.fibonacci import latest_up_leg, retracement_levels
from src.analysis.structure import nearest_confluent_level, support_resistance_levels
from src.strategies.registry import RISK_STOPS
from src.strategies.types import EvalContext, SetupResult


@RISK_STOPS.register("structural_stop")
def structural_stop(entry_price: float, setup: SetupResult, ctx: EvalContext, params: dict) -> float:
    order = params.get("order", 3)
    tolerance_pct = params.get("tolerance_pct", 0.5)
    buffer_pct = params.get("buffer_pct", 0.1)
    max_stop_pct = params.get("max_stop_pct", 5.0)
    fallback_pct = params.get("fallback_pct", 0.5)
    min_confluence = params.get("min_confluence", 1)

    support_levels = support_resistance_levels(
        ctx.marked_window, order=order, tolerance_pct=tolerance_pct, marked=ctx.marked_window
    )["support"]

    fib_levels: list[float] = []
    leg = latest_up_leg(ctx.price_window, order=order, marked=ctx.marked_window)
    if leg is not None:
        fib_levels = list(retracement_levels(leg["low"], leg["high"]).values())

    level = nearest_confluent_level(
        entry_price,
        support_levels + fib_levels,
        confluence_groups=(support_levels, fib_levels),
        direction="below",
        min_confluence=min_confluence,
        tolerance_pct=tolerance_pct,
    )

    # No real structure below entry at all (a void, or not enough history
    # yet) — fall back to a tight, conservative default rather than no
    # stop, same convention as risk/atr_stop.py's own fallback.
    if level is None:
        return entry_price * (1 - fallback_pct / 100)

    stop_price = level * (1 - buffer_pct / 100)

    # The nearest real level can still be unreasonably far below (a genuine
    # structural void) — a stop that wide isn't a usable risk control, so
    # fall back instead of silently accepting an arbitrarily large loss.
    stop_distance_pct = (entry_price - stop_price) / entry_price * 100
    if stop_distance_pct > max_stop_pct:
        return entry_price * (1 - fallback_pct / 100)

    return stop_price
