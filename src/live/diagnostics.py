"""Per-bar diagnostic trace of the strategy's context -> setup ->
confirmations chain, for live observability — the user asked to see, for
every candle that closes, exactly which pieces passed or failed and the
final yes/no, with a full history to click back through (not just "no
signal" the way check_market's own state machine reports it).

Unlike _find_entry_signal (src/backtest/engine.py), which short-circuits
on the first failing piece for backtest speed, `trace_entry_signal`
evaluates EVERY piece regardless of earlier failures. This is purely
observational — it never feeds back into check_market's own state machine
or the actual entry/exit decision, only reports on it after the fact,
same separation opportunity_scan.py/trade_audit.py already keep between
"what the strategy decided" and "measuring/explaining that decision".
"""
from __future__ import annotations

import pandas as pd

from src.live.paper_trading import _evaluate_at
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext


def trace_entry_signal(ctx: EvalContext, strategy: ResolvedStrategy) -> dict:
    """Runs the strategy's full chain on `ctx` without short-circuiting.
    Returns a dict with per-stage pass/fail and the overall `signal`
    (True only if every stage passed) — see module docstring."""
    trend = strategy.context_fn(ctx, strategy.context_params)
    context_passed = strategy.required_context == "any" or trend == strategy.required_context

    setup = strategy.setup_fn(ctx, strategy.setup_params)
    setup_passed = setup is not None

    confirmations = []
    for confirmation in strategy.confirmations:
        result = confirmation.fn(ctx, confirmation.params)
        confirmations.append({"piece": confirmation.name, "passed": result is not None})

    signal = context_passed and setup_passed and all(c["passed"] for c in confirmations)

    return {
        "context_value": trend,
        "context_required": strategy.required_context,
        "context_passed": bool(context_passed),
        "setup_passed": bool(setup_passed),
        "confirmations": confirmations,
        "signal": bool(signal),
    }


def diagnose_entry(
    df: pd.DataFrame,
    higher_tf_df: pd.DataFrame | None,
    strategy: ResolvedStrategy,
    at_index: int | None = None,
    higher_tf_lookback_bars: int = 50,
) -> dict:
    """Traces the entry chain at bar `at_index` (default: the latest bar
    in `df`) — reuses paper_trading._evaluate_at's exact ctx reconstruction
    (same no-lookahead window the live lab itself uses), so this always
    reports on precisely the same information the real decision had.
    Returns `trace_entry_signal`'s dict plus `timestamp`/`close` for the
    evaluated bar."""
    df = df.reset_index(drop=True)
    if at_index is None:
        at_index = len(df) - 1
    ctx, _ = _evaluate_at(df, strategy, higher_tf_df, at_index, higher_tf_lookback_bars)
    trace = trace_entry_signal(ctx, strategy)
    trace["timestamp"] = str(df["timestamp"].iloc[at_index])
    trace["close"] = float(df["close"].iloc[at_index])
    return trace
