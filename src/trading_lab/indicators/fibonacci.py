"""Fibonacci retracement/extension levels over a swing (low, high).

Important: Fibonacci has no physical basis in the market — it works as a
confluence zone because a huge number of participants use it, not because
price is mathematically "forced" to respect it. That's why it's treated
here as a *zone of interest* (a price band), never as a signal on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

RETRACEMENT_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXTENSION_RATIOS = (1.272, 1.414, 1.618, 2.0, 2.618)


@dataclass(frozen=True)
class FibLevels:
    swing_low: float
    swing_high: float
    direction: str  # 'up' (retracement measured from high down) or 'down' (the other way)
    retracements: dict[float, float]
    extensions: dict[float, float]


def compute_fib_levels(swing_low: float, swing_high: float, direction: str) -> FibLevels:
    """Computes retracement/extension levels for the swing [swing_low, swing_high].

    `direction='up'`: the recent move was bullish (low -> high), so
    retracements are measured going down from the high. `direction='down'`:
    the recent move was bearish (high -> low), retracements are measured
    going up from the low.
    """
    if swing_high <= swing_low:
        raise ValueError("swing_high must be greater than swing_low")
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")

    rng = swing_high - swing_low
    retracements: dict[float, float] = {}
    extensions: dict[float, float] = {}

    for ratio in RETRACEMENT_RATIOS:
        if direction == "up":
            retracements[ratio] = swing_high - ratio * rng
        else:
            retracements[ratio] = swing_low + ratio * rng

    for ratio in EXTENSION_RATIOS:
        if direction == "up":
            extensions[ratio] = swing_high - ratio * rng
        else:
            extensions[ratio] = swing_low + ratio * rng

    return FibLevels(
        swing_low=swing_low,
        swing_high=swing_high,
        direction=direction,
        retracements=retracements,
        extensions=extensions,
    )


def confluence_zone(levels: FibLevels, zone_min: float = 0.5, zone_max: float = 0.618) -> tuple[float, float]:
    """Returns the price range [lo, hi] of the confluence zone between two
    retracement ratios (default: the "golden zone" 0.5-0.618)."""
    a = levels.retracements[zone_min]
    b = levels.retracements[zone_max]
    return (min(a, b), max(a, b))


def price_in_zone(price: float, zone: tuple[float, float]) -> bool:
    lo, hi = zone
    return lo <= price <= hi
