"""Confirmation piece: above-average, buyer-dominant volume.

Wraps the existing src.analysis.volume — see docs/PLAN.md, "Config-driven
strategy catalog".
"""
from __future__ import annotations

from src.analysis.volume import confirms_buyer_pressure, directional_volume_bias
from src.strategies.registry import CONFIRMATIONS
from src.strategies.types import ConfirmationResult, EvalContext


@CONFIRMATIONS.register("volume")
def volume(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    window = params.get("window", 20)
    spike_threshold = params.get("spike_threshold", 1.2)
    min_bias = params.get("min_bias", 0.1)

    last_idx = len(ctx.price_window) - 1
    if not confirms_buyer_pressure(
        ctx.price_window, idx=last_idx, window=window, spike_threshold=spike_threshold, min_bias=min_bias
    ):
        return None

    bias = directional_volume_bias(ctx.price_window, window=window)
    return ConfirmationResult(name="volume", extras={"volume_bias": bias})
