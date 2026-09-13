"""Context piece: an explicitly experimental, "just to see results"
alternative to dow_trend/ma_trend — looks up the current bar's ADX/RSI
bucket in a pre-built historical frequency table (see
src.analysis.probability_table) and returns "uptrend" if the empirical
P(up) clears a threshold.

NOT walk-forward validated as shipped: the table is normally built from the
same window being backtested (see src.analysis.probability_table's module
docstring). Treat any result through this piece as exploratory, per
docs/PLAN.md.

The table itself isn't a param (YAML can't embed a DataFrame) — it's built
once by the caller (see scripts/run_lab_comparison.py) and passed in via
`params["table"]`.
"""
from __future__ import annotations

from src.analysis.adx import adx
from src.analysis.momentum import rsi
from src.analysis.probability_table import lookup_probability
from src.strategies.registry import CONTEXT
from src.strategies.types import EvalContext


@CONTEXT.register("probability_trend")
def probability_trend(ctx: EvalContext, params: dict) -> str:
    table = params.get("table")
    if table is None:
        raise ValueError("probability_trend requires a pre-built 'table' param (see probability_table.py)")

    min_probability = params.get("min_probability", 0.55)
    adx_period = params.get("adx_period", 14)
    rsi_period = params.get("rsi_period", 14)

    last_adx = adx(ctx.price_window, period=adx_period).iloc[-1]
    last_rsi = rsi(ctx.price_window["close"], period=rsi_period).iloc[-1]

    probability = lookup_probability(table, last_adx, last_rsi)
    if probability is None:
        return "sideways"  # unknown bucket / not enough history -> no opinion
    return "uptrend" if probability >= min_probability else "sideways"
