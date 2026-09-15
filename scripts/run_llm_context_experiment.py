"""Does a local LLM's read of the recent 5-minute price action correlate
with how a trade actually turned out? Strictly a post-hoc check — the
score never touches the entry decision (see src/analysis/llm_score.py's
docstring on why).

For every trade the frozen baseline (trend_pullback_htf_confluence) took
in the known window, takes the `n_candles` 5-minute candles immediately
before the entry, asks the local Ollama model for a 0-100 score, and
records it alongside the trade's actual R-multiple outcome. Then checks:
does a higher score associate with a better outcome (Spearman correlation
+ a simple high/low split), or not?

Requires a local Ollama server running (see docs/PLAN.md's live-lab notes)
— .ollama-local/bin/ollama serve, with qwen2.5:7b-instruct pulled.

Usage:
    .venv/bin/python scripts/run_llm_context_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

from config.markets import MARKETS
from src.analysis.llm_score import score_candles
from src.backtest.engine import run_backtest
from src.backtest.experiment import build_trade_log
from src.data.fetcher import fetch_ohlcv
from src.report.paths import REPORTS_DIR
from src.strategies.builder import load_strategy

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"
LOWER_TIMEFRAME = "5m"
SINCE, UNTIL = "2023-09-13", "2026-09-13"
N_CANDLES = 24  # ~2 hours of 5-min context ("puede ser varias horas atrás")

STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)

    logs = []
    lower_tf_by_market: dict[str, pd.DataFrame] = {}
    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=SINCE, until=UNTIL)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)
        lower_tf_by_market[market.symbol] = fetch_ohlcv(market.symbol, LOWER_TIMEFRAME, since=SINCE, until=UNTIL)
        trades, _ = run_backtest(df, strategy, higher_tf_df=higher_tf_df)
        log = build_trade_log(trades, market=market.symbol, tf_execution=TIMEFRAME)
        if not log.empty:
            logs.append(log)

    full_log = pd.concat(logs, ignore_index=True)
    print(f"Scoring {len(full_log)} trades with the local LLM (this takes a few minutes)...")

    scores = []
    for _, trade in full_log.iterrows():
        lower_tf = lower_tf_by_market[trade["market"]]
        window = lower_tf[lower_tf["timestamp"] < trade["entry_time"]].tail(N_CANDLES)
        scores.append(score_candles(window) if len(window) == N_CANDLES else None)
    full_log["llm_score"] = scores

    n_scored = full_log["llm_score"].notna().sum()
    print(f"\nScored: {n_scored} / {len(full_log)} (missing = server error or not enough 5m history)")

    scored = full_log.dropna(subset=["llm_score"])
    if len(scored) >= 5:
        corr, p_value = spearmanr(scored["llm_score"], scored["r_multiple"])
        median_score = scored["llm_score"].median()
        high = scored[scored["llm_score"] > median_score]
        low = scored[scored["llm_score"] <= median_score]
        print(f"\nSpearman correlation (llm_score vs. r_multiple): {corr:.3f} (p={p_value:.3f})")
        print(f"median score: {median_score}")
        print(f"  above-median score: n={len(high)}, avg R={high['r_multiple'].mean():.3f}, "
              f"win rate={100 * (high['r_multiple'] > 0).mean():.1f}%")
        print(f"  at/below-median score: n={len(low)}, avg R={low['r_multiple'].mean():.3f}, "
              f"win rate={100 * (low['r_multiple'] > 0).mean():.1f}%")
    else:
        print("Too few scored trades to compute a correlation.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "llm_context_trade_log.csv"
    full_log.to_csv(out_path, index=False)
    print(f"\nFull annotated trade log saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
