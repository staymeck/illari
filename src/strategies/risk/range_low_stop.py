"""Risk piece: places the stop just below the accumulation range's own
low — the Wyckoff reading is that this level held as support while the
range built up, so a break back below it means the accumulation read was
wrong, not just ordinary noise. Reads `range_low` directly from the
setup's extras (wyckoff_accumulation_breakout.py) rather than
re-detecting the range independently — stop_fn receives the setup, so
there's no need to recompute what it already found.
"""
from __future__ import annotations

from src.strategies.registry import RISK_STOPS
from src.strategies.types import EvalContext, SetupResult


@RISK_STOPS.register("range_low_stop")
def range_low_stop(entry_price: float, setup: SetupResult, ctx: EvalContext, params: dict) -> float:
    buffer_pct = params.get("buffer_pct", 0.5)
    fallback_pct = params.get("fallback_pct", 1.0)

    range_low = setup.extras.get("range_low") if setup is not None else None
    if range_low is None:
        return entry_price * (1 - fallback_pct / 100)

    return range_low * (1 - buffer_pct / 100)
