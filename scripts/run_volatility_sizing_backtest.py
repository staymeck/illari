"""Backtest-simulates volatility-regime-scaled position sizing against the
exact same trade sequence the frozen baseline (trend_pullback_htf_confluence,
see scripts/run_baseline_experiment.py) already produced — same signals,
same entries/exits/stops/targets, only how much of the account's equity
each trade risks changes. Per the project's own rule: never deploy
untested (docs/PLAN.md's staged experiment protocol) — mirrors
scripts/run_confluence_sizing_backtest.py exactly, swapping the sizing
driver from stop-level confluence to src.analysis.regime.volatility_percentile
at each trade's entry bar (see src/live/position_sizing.py for why this
idea, and why it couldn't be evaluated on the branch it was rescued from).

Usage:
    .venv/bin/python scripts/run_volatility_sizing_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.analysis.regime import volatility_percentile
from src.backtest.engine import run_backtest
from src.live.position_sizing import volatility_size_multiplier
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
WINDOWS = [("known", "2023-09-13", "2026-09-13"), ("virgin", "2020-09-01", "2023-09-13")]
STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"

# Sizing schemes to compare. "uniform" is the control: current live
# behavior (100% of equity per trade). The others scale by
# volatility_size_multiplier at a few (min_mult, max_mult) settings.
SCHEMES: dict[str, tuple[float, float] | None] = {
    "uniform": None,
    "volatility_default": (0.5, 1.0),
    "volatility_aggressive": (0.25, 1.0),
    "volatility_mild": (0.75, 1.0),
}

ATR_PERIOD = 14
ATR_LOOKBACK = 100


def _from_data(fetch_ohlcv, market, since, until):
    df = fetch_ohlcv(market.symbol, TIMEFRAME, since=since, until=until).reset_index(drop=True)
    higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=since, until=until)
    return df, higher_tf_df


def _percentile_for_trade(df: pd.DataFrame, trade: pd.Series) -> float:
    matches = df.index[df["timestamp"] == trade["entry_time"]]
    if len(matches) == 0:
        return float("nan")
    entry_idx = int(matches[0])
    # Volatility as measured using only bars up to and including the
    # signal bar (entry_idx - 1 is the last closed bar when the decision
    # was made) - no lookahead.
    signal_idx = entry_idx - 1
    if signal_idx < 0:
        return float("nan")
    window = df.iloc[: signal_idx + 1]
    return volatility_percentile(window, atr_period=ATR_PERIOD, lookback=ATR_LOOKBACK)


def _simulate_equity(trades: pd.DataFrame, bounds: tuple[float, float] | None, initial_equity: float = 10000.0) -> dict:
    equity = initial_equity
    peak = initial_equity
    max_drawdown_pct = 0.0
    for _, trade in trades.sort_values("entry_time").iterrows():
        if bounds is None:
            m = 1.0
        else:
            min_mult, max_mult = bounds
            m = volatility_size_multiplier(trade["vol_percentile"], min_mult=min_mult, max_mult=max_mult)
        equity = equity * (1 + m * trade["pnl_pct"] / 100)
        peak = max(peak, equity)
        drawdown_pct = (peak - equity) / peak * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    total_return_pct = (equity - initial_equity) / initial_equity * 100
    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_drawdown_pct}


def main() -> None:
    from src.data.fetcher import fetch_ohlcv

    strategy = load_strategy(STRATEGY_PATH)
    all_trades = []

    for label, since, until in WINDOWS:
        for market in MARKETS:
            df, higher_tf_df = _from_data(fetch_ohlcv, market, since, until)
            trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
            if trades.empty:
                continue
            trades = trades.copy()
            trades["market"] = market.symbol
            trades["window"] = label
            trades["vol_percentile"] = trades.apply(lambda t: _percentile_for_trade(df, t), axis=1)
            all_trades.append(trades)

    full = pd.concat(all_trades, ignore_index=True)

    print(f"VOLATILITY-REGIME-SCALED POSITION SIZING — backtest against the frozen baseline sequence ({strategy.name})\n")
    print(f"n_trades: {len(full)}\n")

    pct = full["vol_percentile"].dropna()
    print("Volatility percentile distribution at entry (0=calmest recent conditions, 100=most volatile):")
    print(f"  n_valid: {len(pct)}/{len(full)}")
    if not pct.empty:
        print(f"  mean={pct.mean():.1f}  median={pct.median():.1f}  p25={pct.quantile(0.25):.1f}  p75={pct.quantile(0.75):.1f}")
    print()

    print(f"{'scheme':<26}{'market':<12}{'window':<8}{'total_return_pct':>18}{'max_drawdown_pct':>18}")
    for scheme_name, bounds in SCHEMES.items():
        for (market, window), group in full.groupby(["market", "window"]):
            result = _simulate_equity(group, bounds)
            print(
                f"{scheme_name:<26}{market:<12}{window:<8}"
                f"{result['total_return_pct']:>18.2f}{result['max_drawdown_pct']:>18.2f}"
            )
        print()

    print("Average across markets/windows (mean of independent per-market outcomes, NOT a pooled equity curve):")
    for scheme_name, bounds in SCHEMES.items():
        returns, drawdowns = [], []
        for (_market, _window), group in full.groupby(["market", "window"]):
            result = _simulate_equity(group, bounds)
            returns.append(result["total_return_pct"])
            drawdowns.append(result["max_drawdown_pct"])
        avg_return = sum(returns) / len(returns)
        avg_dd = sum(drawdowns) / len(drawdowns)
        print(f"  {scheme_name:<26} avg_total_return: {avg_return:>8.2f}%   avg_max_drawdown: {avg_dd:>8.2f}%")


if __name__ == "__main__":
    main()
