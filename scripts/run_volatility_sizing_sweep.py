"""Parameter sweep for volatility-regime position sizing (see
src/live/position_sizing.py, scripts/run_volatility_sizing_backtest.py),
before considering it for live deployment — the first pass only tried 3
(min_mult, max_mult) points; this widens both the sizing-bound grid and
the volatility-measurement window itself (atr_period, lookback), to check
the result isn't an artifact of the first 3 arbitrary points chosen.

Fetches each market/window's data ONCE, then evaluates every
(atr_period, lookback) x min_mult combination against it — the backtest
signals/entries/exits themselves never change (same frozen baseline
sequence), only how the volatility percentile is measured and how it maps
to size. `max_mult` is fixed at 1.0 throughout (spot, no leverage — this
sizing scheme was never about betting MORE than the uniform baseline,
only about betting less when it's risky).

Usage:
    .venv/bin/python scripts/run_volatility_sizing_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.analysis.regime import volatility_percentile
from src.backtest.engine import run_backtest
from src.data.fetcher import fetch_ohlcv
from src.live.position_sizing import volatility_size_multiplier
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]
STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"

# (atr_period, lookback) pairs to measure the volatility percentile with.
PERCENTILE_CONFIGS = [(14, 50), (14, 100), (14, 200), (20, 100)]
MIN_MULTS = [0.1, 0.25, 0.4, 0.5, 0.65, 0.75, 0.9]
MAX_MULT = 1.0


def _percentile_for_trade(df: pd.DataFrame, trade: pd.Series, atr_period: int, lookback: int) -> float:
    matches = df.index[df["timestamp"] == trade["entry_time"]]
    if len(matches) == 0:
        return float("nan")
    entry_idx = int(matches[0])
    signal_idx = entry_idx - 1
    if signal_idx < 0:
        return float("nan")
    window = df.iloc[: signal_idx + 1]
    return volatility_percentile(window, atr_period=atr_period, lookback=lookback)


def _simulate_equity(trades: pd.DataFrame, pct_col: str, min_mult: float, initial_equity: float = 10000.0) -> dict:
    equity = initial_equity
    peak = initial_equity
    max_drawdown_pct = 0.0
    for _, trade in trades.sort_values("entry_time").iterrows():
        m = volatility_size_multiplier(trade[pct_col], min_mult=min_mult, max_mult=MAX_MULT)
        equity = equity * (1 + m * trade["pnl_pct"] / 100)
        peak = max(peak, equity)
        drawdown_pct = (peak - equity) / peak * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    total_return_pct = (equity - initial_equity) / initial_equity * 100
    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_drawdown_pct}


def _simulate_uniform(trades: pd.DataFrame, initial_equity: float = 10000.0) -> dict:
    equity = initial_equity
    peak = initial_equity
    max_drawdown_pct = 0.0
    for _, trade in trades.sort_values("entry_time").iterrows():
        equity = equity * (1 + trade["pnl_pct"] / 100)
        peak = max(peak, equity)
        max_drawdown_pct = max(max_drawdown_pct, (peak - equity) / peak * 100)
    return {"total_return_pct": (equity - initial_equity) / initial_equity * 100, "max_drawdown_pct": max_drawdown_pct}


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    all_trades = []

    for label, since, until in WINDOWS:
        for market in MARKETS:
            df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until).reset_index(drop=True)
            higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            if trades.empty:
                continue
            trades = trades.copy()
            trades["market"] = market.symbol
            trades["window"] = label
            for atr_period, lookback in PERCENTILE_CONFIGS:
                col = f"vol_pct_{atr_period}_{lookback}"
                trades[col] = trades.apply(lambda t: _percentile_for_trade(df, t, atr_period, lookback), axis=1)
            all_trades.append(trades)

    full = pd.concat(all_trades, ignore_index=True)
    groups = list(full.groupby(["market", "window"]))

    print(f"VOLATILITY SIZING PARAMETER SWEEP — {strategy.name}, n_trades={len(full)}\n")

    uniform_returns, uniform_dds = [], []
    for _, group in groups:
        r = _simulate_uniform(group)
        uniform_returns.append(r["total_return_pct"])
        uniform_dds.append(r["max_drawdown_pct"])
    uniform_avg_return = sum(uniform_returns) / len(uniform_returns)
    uniform_avg_dd = sum(uniform_dds) / len(uniform_dds)
    print(f"uniform (control): avg_total_return={uniform_avg_return:.2f}%  avg_max_drawdown={uniform_avg_dd:.2f}%\n")

    print(f"{'atr_period':>10}{'lookback':>10}{'min_mult':>10}{'avg_return%':>14}{'avg_drawdown%':>16}{'beats_uniform_on_both':>24}")
    rows = []
    for atr_period, lookback in PERCENTILE_CONFIGS:
        col = f"vol_pct_{atr_period}_{lookback}"
        for min_mult in MIN_MULTS:
            returns, dds = [], []
            for _, group in groups:
                r = _simulate_equity(group, col, min_mult)
                returns.append(r["total_return_pct"])
                dds.append(r["max_drawdown_pct"])
            avg_return = sum(returns) / len(returns)
            avg_dd = sum(dds) / len(dds)
            beats_both = avg_return > uniform_avg_return and avg_dd < uniform_avg_dd
            rows.append(
                {"atr_period": atr_period, "lookback": lookback, "min_mult": min_mult,
                 "avg_return": avg_return, "avg_dd": avg_dd, "beats_both": beats_both}
            )
            print(f"{atr_period:>10}{lookback:>10}{min_mult:>10.2f}{avg_return:>14.2f}{avg_dd:>16.2f}{str(beats_both):>24}")

    sweep_df = pd.DataFrame(rows)
    n_beat_both = int(sweep_df["beats_both"].sum())
    print(f"\n{n_beat_both}/{len(sweep_df)} combinations beat uniform on BOTH avg return and avg drawdown.")

    best = sweep_df.sort_values("avg_return", ascending=False).iloc[0]
    print(f"\nBest avg_return combo: atr_period={int(best['atr_period'])} lookback={int(best['lookback'])} "
          f"min_mult={best['min_mult']:.2f} -> avg_return={best['avg_return']:.2f}% avg_dd={best['avg_dd']:.2f}%")

    best_dd = sweep_df.sort_values("avg_dd").iloc[0]
    print(f"Best avg_drawdown combo: atr_period={int(best_dd['atr_period'])} lookback={int(best_dd['lookback'])} "
          f"min_mult={best_dd['min_mult']:.2f} -> avg_return={best_dd['avg_return']:.2f}% avg_dd={best_dd['avg_dd']:.2f}%")


if __name__ == "__main__":
    main()
