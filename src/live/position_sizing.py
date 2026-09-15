"""Position sizing scaled by market-relative volatility regime.

Rescued and re-derived from a colleague's exploratory branch
(`probabilities-to-into-trade`, David A — not part of this lab's own
history): its `indicators/volatility.py` scaled BOTH stop distance and
position size by current ATR vs. its own recent moving-average ratio.
The stop-distance half is already covered here by
`src/strategies/risk/regime_adaptive_stop.py` (via
`src.analysis.regime.volatility_percentile` — a percentile rank, which is
scale-free and bounded 0-100 regardless of a market's own ATR distribution
shape, preferred here over that branch's raw ratio-to-moving-average for
the same reason `regime_adaptive_stop.py` already chose it). What's new
here is the position-SIZE half, which this project never had: risk less
capital when volatility (and uncertainty) is elevated, the normal amount
when the market is calm — deliberately grounded in a measured, objective
market condition (Kaufman's "Version A" adaptation), never the strategy's
own recent win/loss streak ("Version B" — see src/analysis/regime.py's
module docstring for why that distinction matters).

That colleague's own branch never actually validated this against a
strategy with a real edge — its own confluence/breakout variants returned
-34% to -100%, so no conclusion about the sizing idea itself could be
drawn from those numbers (garbage in, garbage out). This module is
retested from scratch against THIS project's own frozen baseline trade
sequence (see scripts/run_volatility_sizing_backtest.py) before any live
use, same rule as everything else in this lab.
"""
from __future__ import annotations

import pandas as pd

# Fraction of equity risked at the calmest (pct=0) and most volatile
# (pct=100) ends of the market's own recent volatility distribution.
# Never above 1.0 - this is spot, no leverage, so "calmer than usual" can
# only mean "risk the normal full amount," never more than you have.
DEFAULT_MIN_MULT = 0.5
DEFAULT_MAX_MULT = 1.0


def volatility_size_multiplier(
    percentile: float | None, min_mult: float = DEFAULT_MIN_MULT, max_mult: float = DEFAULT_MAX_MULT
) -> float:
    """Linearly maps a volatility_percentile reading (0-100, see
    src.analysis.regime.volatility_percentile) to a fraction of equity to
    risk: `max_mult` at percentile 0 (calmest recent conditions),
    `min_mult` at percentile 100 (most volatile), linear in between.

    `percentile=None`/NaN (not enough history yet) is treated as the
    middle of the distribution (50) rather than an extreme in either
    direction — same "assume unknown is average" convention already used
    by regime_adaptive_stop.py, instead of silently under- or over-sizing
    on missing data."""
    if percentile is None or pd.isna(percentile):
        percentile = 50.0
    percentile = min(max(percentile, 0.0), 100.0)
    return max_mult - (max_mult - min_mult) * (percentile / 100.0)
