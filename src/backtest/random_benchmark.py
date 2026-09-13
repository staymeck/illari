"""Monte Carlo random-entry benchmark: places entries at uniformly random
bars — no context/setup/confirmations at all — with the exact same risk
mechanics as a real strategy (same stop/target functions, same fees, same
max holding bars, same "one trade at a time" rule). The only thing that
differs from a real backtest is *when* we enter, not how a position is
managed once open.

Why this exists: it's the null hypothesis every result in this lab should
be checked against. If a strategy's return doesn't clearly stand out from
a large number of random draws with the same trade count, its filtering
isn't demonstrated to add value beyond chance — see docs/PLAN.md and the
session discussion that led to this (walk-forward showed the SOL "edge"
decaying, raising exactly this question for every other result too).

`entry_prob` is calibrated so a single trial fires *approximately*
`target_n_trades` times — exact counts vary trial to trial (as they should:
this is a stochastic process), and `run_monte_carlo` runs many trials to
build a distribution rather than relying on one.
"""
from __future__ import annotations

import random

import pandas as pd

from src.backtest.engine import _build_trade_record, _walk_to_exit, compute_metrics
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext, SetupResult


def run_random_trial(
    df: pd.DataFrame, strategy: ResolvedStrategy, entry_prob: float, rng: random.Random
) -> tuple[pd.DataFrame, pd.Series]:
    """One random-entry simulation over `df`, using `strategy`'s risk pieces
    (stop/target functions, fees, holding period) unchanged — only the entry
    timing is random instead of rule-based."""
    df = df.reset_index(drop=True)
    trades: list[dict] = []
    equity = strategy.initial_equity
    equity_curve: list[float] = [equity]

    i = strategy.lookback_bars
    n = len(df)
    while i < n - 1:
        price_window = df.iloc[max(0, i + 1 - strategy.lookback_bars) : i + 1]

        if len(price_window) < strategy.lookback_bars or rng.random() >= entry_prob:
            equity_curve.append(equity)
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = df["open"].iloc[entry_idx]

        # No setup here — a random entry has no support/breakout/band level
        # to anchor a stop to, so the entry price itself is the reference.
        setup = SetupResult(reference_level=entry_price)
        ctx = EvalContext(price_window=price_window, marked_window=pd.DataFrame())
        stop_price = strategy.stop_fn(entry_price, setup, ctx, strategy.stop_params)
        risk = entry_price - stop_price
        if risk <= 0:
            equity_curve.append(equity)
            i += 1
            continue
        target_price = strategy.target_fn(entry_price, stop_price, ctx, strategy.target_params)

        exit_idx, exit_price, exit_reason, stop_was_premature = _walk_to_exit(
            df, entry_idx, stop_price, target_price, strategy.stop_trigger, strategy.max_holding_bars
        )
        trade, equity = _build_trade_record(
            df, entry_idx, exit_idx, entry_price, exit_price, exit_reason, stop_was_premature,
            stop_price, target_price, strategy.fee_pct, equity, extras={},
        )
        trades.append(trade)
        equity_curve.append(equity)

        i = exit_idx + 1  # don't overlap trades, same rule as run_backtest

    return pd.DataFrame(trades), pd.Series(equity_curve, name="equity")


def run_monte_carlo(
    df: pd.DataFrame,
    strategy: ResolvedStrategy,
    target_n_trades: int,
    n_simulations: int = 500,
    seed: int | None = None,
) -> pd.DataFrame:
    """Runs `n_simulations` random trials calibrated to fire roughly
    `target_n_trades` times each, and returns one row of `compute_metrics`
    output per simulation — the null distribution to compare a real
    strategy's result against."""
    df = df.reset_index(drop=True)
    eligible_bars = max(len(df) - strategy.lookback_bars - 1, 1)
    entry_prob = min(target_n_trades / eligible_bars, 1.0) if target_n_trades > 0 else 0.0

    rng = random.Random(seed)
    rows = []
    for _ in range(n_simulations):
        trades, equity_curve = run_random_trial(df, strategy, entry_prob, rng)
        rows.append(compute_metrics(trades, equity_curve, strategy.initial_equity))
    return pd.DataFrame(rows)


def percentile_rank(real_value: float, simulated_values: pd.Series | list[float]) -> float:
    """Where `real_value` falls among `simulated_values`, from 0 (worse than
    every simulation) to 100 (better than every simulation). Ties count as
    "at or below" — e.g. matching the median of a symmetric distribution
    lands at 50, not higher."""
    simulated_values = pd.Series(simulated_values)
    if simulated_values.empty:
        raise ValueError("simulated_values must not be empty")
    return float((simulated_values <= real_value).mean() * 100)
