"""Ties together the three things agreed after walk-forward: for one
(symbol, timeframe, date range), runs the deterministic strategy, the
probabilistic one, and a random-entry Monte Carlo benchmark, side by side —
prints a comparison table (including where each real strategy ranks against
500 random draws) and writes one combined entries chart. See docs/PLAN.md.

Usage:
    .venv/bin/python scripts/run_lab_comparison.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.probability_table import build_frequency_table
from src.backtest.engine import compute_metrics, run_backtest
from src.backtest.random_benchmark import percentile_rank, run_monte_carlo, run_random_trial
from src.data.fetcher import fetch_ohlcv
from src.report.chart import write_entries_chart
from src.report.render import write_run_report
from src.strategies.builder import load_strategy

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE = "2023-09-13"
UNTIL = "2026-09-13"
N_SIMULATIONS = 500
SEED = 42


def main() -> None:
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df)} candles\n")

    # (a) Deterministic
    det_strategy = load_strategy("config/strategies/trend_pullback_fib.yaml")
    det_trades, det_equity = run_backtest(df, det_strategy)
    det_metrics = compute_metrics(det_trades, det_equity, det_strategy.initial_equity)
    print(f"Deterministic ({det_strategy.name}): {det_metrics}")

    # (b) Probabilistic — table built in-sample over the same window (see
    # src/analysis/probability_table.py's honesty caveat: not yet
    # walk-forward validated).
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)
    prob_strategy = load_strategy("config/strategies/probability_pullback.yaml")
    prob_strategy.context_params["table"] = table
    prob_trades, prob_equity = run_backtest(df, prob_strategy)
    prob_metrics = compute_metrics(prob_trades, prob_equity, prob_strategy.initial_equity)
    print(f"Probabilistic ({prob_strategy.name}): {prob_metrics}")

    # (c) Random Monte Carlo benchmark, calibrated to the deterministic
    # strategy's trade count, plus one representative trial for the chart.
    mc = run_monte_carlo(df, det_strategy, target_n_trades=det_metrics["n_trades"], n_simulations=N_SIMULATIONS, seed=SEED)
    det_rank = percentile_rank(det_metrics["total_return_pct"], mc["total_return_pct"])
    prob_rank = percentile_rank(prob_metrics["total_return_pct"], mc["total_return_pct"])
    print(f"\nRandom benchmark ({N_SIMULATIONS} sims, ~{det_metrics['n_trades']} trades each):")
    print(mc[["n_trades", "total_return_pct", "win_rate_pct", "profit_factor"]].describe())
    print(f"\nDeterministic total_return_pct percentile vs random: {det_rank:.1f}")
    print(f"Probabilistic total_return_pct percentile vs random: {prob_rank:.1f}")

    entry_prob = det_metrics["n_trades"] / max(len(df) - det_strategy.lookback_bars - 1, 1)
    random_trades, _ = run_random_trial(df, det_strategy, entry_prob=entry_prob, rng=random.Random(SEED))

    # Reports + combined chart
    write_run_report(SYMBOL, TIMEFRAME, det_trades, det_metrics, scenario="lab_comparison_deterministic")
    write_run_report(SYMBOL, TIMEFRAME, prob_trades, prob_metrics, scenario="lab_comparison_probabilistic")
    chart_path = write_entries_chart(
        df,
        {"deterministic": det_trades, "probabilistic": prob_trades, "random": random_trades},
        SYMBOL,
        TIMEFRAME,
    )
    print(f"\nCombined entries chart: {chart_path}")


if __name__ == "__main__":
    main()
