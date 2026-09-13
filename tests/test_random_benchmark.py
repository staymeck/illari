"""Tests for src/backtest/random_benchmark.py."""
import random

import pandas as pd

from src.backtest.random_benchmark import percentile_rank, run_monte_carlo, run_random_trial
from src.strategies.builder import load_strategy


def _flat_df(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
        }
    )


def test_run_random_trial_produces_no_trades_at_zero_probability():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    trades, equity_curve = run_random_trial(_flat_df(), strategy, entry_prob=0.0, rng=random.Random(0))

    assert trades.empty
    assert (equity_curve == strategy.initial_equity).all()


def test_run_random_trial_produces_trades_at_full_probability():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    trades, _ = run_random_trial(_flat_df(), strategy, entry_prob=1.0, rng=random.Random(0))

    assert len(trades) > 0
    # Trades must not overlap: each entry strictly after the previous exit.
    for k in range(1, len(trades)):
        assert trades.iloc[k]["entry_time"] > trades.iloc[k - 1]["exit_time"]


def test_run_monte_carlo_average_trade_count_matches_target():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    df = _flat_df(n=1000)

    results = run_monte_carlo(df, strategy, target_n_trades=10, n_simulations=200, seed=42)

    # Individual trials vary (it's stochastic), but the average across many
    # trials should land reasonably close to the target.
    assert 5 <= results["n_trades"].mean() <= 15


def test_run_monte_carlo_is_reproducible_with_a_seed():
    strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    df = _flat_df(n=500)

    a = run_monte_carlo(df, strategy, target_n_trades=5, n_simulations=50, seed=7)
    b = run_monte_carlo(df, strategy, target_n_trades=5, n_simulations=50, seed=7)

    assert a["n_trades"].tolist() == b["n_trades"].tolist()


def test_percentile_rank_known_values():
    simulated = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile_rank(5.5, simulated) == 100.0
    assert percentile_rank(0.5, simulated) == 0.0
    assert percentile_rank(3.0, simulated) == 60.0  # 3 of 5 values <= 3.0


def test_percentile_rank_rejects_empty_simulations():
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        percentile_rank(1.0, [])
