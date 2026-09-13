"""Confirmation piece: price trading above rolling VWAP (bullish bias) —
wraps src.analysis.vwap.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from __future__ import annotations

import pandas as pd

from src.analysis.vwap import vwap_bias as compute_vwap_bias
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("vwap_bias")
def vwap_bias(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    window = params.get("window", 20)
    min_bias_pct = params.get("min_bias_pct", 0.0)

    bias = compute_vwap_bias(ctx.price_window, window=window)
    if pd.isna(bias) or bias < min_bias_pct:
        return None
    return ConfirmationResult(name="vwap_bias", extras={"vwap_bias_pct": bias})
