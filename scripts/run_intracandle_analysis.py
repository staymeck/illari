"""Intra-candle microstructure mining: for every 1h candle in the known
window (2023-09-13 to 2026-09-13 — reusing the 5-minute data already
cached from the earlier 5m-timeframe tests, instead of a fresh multi-year
download), computes efficiency_ratio and direction_agreement_pct from its
5-minute sub-candles (src/analysis/intracandle.py), and compares candles
where trend_pullback_htf_minimal actually entered against the rest of the
market.

Pure data mining, explicitly BEFORE deciding whether to apply any of it —
see docs/PLAN.md / Bitácora Illari, motivated by the "el mercado no se
mueve en pasos discretos" discussion.

Writes two reports:
  - reports/intracandle_mining_technical.md   (full numbers, for review)
  - reports/intracandle_mining_visual.html    (histograms + summary charts)

Usage:
    .venv/bin/python scripts/run_intracandle_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.markets import MARKETS
from src.analysis.intracandle import compare_groups, compute_intracandle_metrics, lag1_autocorrelation
from src.backtest.engine import run_backtest
from src.data.fetcher import fetch_ohlcv
from src.report.paths import REPORTS_DIR
from src.strategies.builder import load_strategy

PARENT_TIMEFRAME = "1h"
CHILD_TIMEFRAME = "5m"
HIGHER_TIMEFRAME = "1d"
# Reuses the already-cached 3-year window from the earlier 5m tests, instead
# of downloading 5-minute data back to 2020 — this is exploratory mining,
# not a validated backtest, so there's no holdout to protect here.
SINCE, UNTIL = "2023-09-13", "2026-09-13"

STRATEGY_PATH = "config/strategies/trend_pullback_htf_minimal.yaml"


def _analyze_market(market, strategy) -> dict:
    parent_df = fetch_ohlcv(market.symbol, PARENT_TIMEFRAME, since=SINCE, until=UNTIL)
    child_df = fetch_ohlcv(market.symbol, CHILD_TIMEFRAME, since=SINCE, until=UNTIL)
    higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=SINCE, until=UNTIL)

    real_trades, _ = run_backtest(parent_df, strategy, higher_tf_df=higher_tf_df)
    metrics = compute_intracandle_metrics(parent_df, child_df)
    is_entered = metrics["timestamp"].isin(real_trades["entry_time"]) if not real_trades.empty else pd.Series(False, index=metrics.index)

    child_returns = child_df["close"].pct_change() * 100

    return {
        "market": market.symbol,
        "n_parent_candles": len(metrics),
        "n_entered": int(is_entered.sum()),
        "efficiency": compare_groups(metrics, is_entered, "efficiency_ratio"),
        "direction_agreement": compare_groups(metrics, is_entered, "direction_agreement_pct"),
        "lag1_autocorr_5m_returns": lag1_autocorrelation(child_returns),
        "metrics_df": metrics,
        "is_entered": is_entered,
    }


def _write_technical_report(results: list[dict], out_path: Path) -> None:
    lines = [
        "# Intra-candle microstructure mining — technical report",
        "",
        f"Window: {SINCE} -> {UNTIL} | parent timeframe: {PARENT_TIMEFRAME} | "
        f"child timeframe: {CHILD_TIMEFRAME} | reference strategy: `{STRATEGY_PATH}`",
        "",
        "Pure data mining, done BEFORE deciding whether to apply any of this to a strategy.",
        "efficiency_ratio: net candle move / sum of absolute 5-min step changes (Kaufman ch.17",
        "Efficiency Ratio, applied at finer resolution). 1.0 = clean, one-directional move;",
        "near 0 = mostly back-and-forth noise. direction_agreement_pct: % of 5-min sub-candles",
        "whose own direction matches the parent candle's overall direction.",
        "",
        "\"entered\" = the 1h candle where trend_pullback_htf_minimal actually opened a trade.",
        "\"rest\" = every other 1h candle in the window. p-value is a two-sided Mann-Whitney U",
        "test (nonparametric) comparing the two groups' distributions.",
        "",
    ]

    for r in results:
        eff, dirn = r["efficiency"], r["direction_agreement"]
        lines += [
            f"## {r['market']}",
            "",
            f"- parent candles: {r['n_parent_candles']} | entered: {r['n_entered']}",
            f"- lag-1 autocorrelation of 5-min returns (whole window): {r['lag1_autocorr_5m_returns']}",
            "",
            "| metric | mean (entered) | median (entered) | mean (rest) | median (rest) | p-value |",
            "|---|---|---|---|---|---|",
            f"| efficiency_ratio | {eff['mean_member']} | {eff['median_member']} | "
            f"{eff['mean_rest']} | {eff['median_rest']} | {eff['p_value']} |",
            f"| direction_agreement_pct | {dirn['mean_member']} | {dirn['median_member']} | "
            f"{dirn['mean_rest']} | {dirn['median_rest']} | {dirn['p_value']} |",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_visual_report(results: list[dict], out_path: Path) -> None:
    markets = [r["market"] for r in results]

    summary = make_subplots(rows=1, cols=2, subplot_titles=("mean efficiency_ratio", "mean direction_agreement_pct"))
    summary.add_trace(go.Bar(x=markets, y=[r["efficiency"]["mean_member"] for r in results], name="entered", marker_color="#2ca02c"), row=1, col=1)
    summary.add_trace(go.Bar(x=markets, y=[r["efficiency"]["mean_rest"] for r in results], name="rest", marker_color="#7f7f7f"), row=1, col=1)
    summary.add_trace(go.Bar(x=markets, y=[r["direction_agreement"]["mean_member"] for r in results], name="entered", marker_color="#2ca02c", showlegend=False), row=1, col=2)
    summary.add_trace(go.Bar(x=markets, y=[r["direction_agreement"]["mean_rest"] for r in results], name="rest", marker_color="#7f7f7f", showlegend=False), row=1, col=2)
    summary.update_layout(title="Entered candles vs. the rest of the market — summary across all 5 markets", barmode="group", template="plotly_white")

    detail = make_subplots(rows=len(results), cols=1, subplot_titles=[f"{r['market']} — efficiency_ratio distribution" for r in results])
    for i, r in enumerate(results, start=1):
        m = r["metrics_df"]
        entered_vals = m.loc[r["is_entered"], "efficiency_ratio"].dropna()
        rest_vals = m.loc[~r["is_entered"], "efficiency_ratio"].dropna()
        detail.add_trace(go.Histogram(x=rest_vals, name="rest", marker_color="#7f7f7f", opacity=0.6, histnorm="probability", nbinsx=30, showlegend=(i == 1)), row=i, col=1)
        detail.add_trace(go.Histogram(x=entered_vals, name="entered", marker_color="#2ca02c", opacity=0.6, histnorm="probability", nbinsx=30, showlegend=(i == 1)), row=i, col=1)
    detail.update_layout(barmode="overlay", template="plotly_white", height=300 * len(results), title="Per-market efficiency_ratio distribution: entered candles vs. the rest")

    html = (
        "<title>Intra-candle mining</title>"
        + summary.to_html(include_plotlyjs="cdn", full_html=False)
        + detail.to_html(include_plotlyjs=False, full_html=False)
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    results = [_analyze_market(market, strategy) for market in MARKETS]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    technical_path = REPORTS_DIR / "intracandle_mining_technical.md"
    visual_path = REPORTS_DIR / "intracandle_mining_visual.html"

    _write_technical_report(results, technical_path)
    _write_visual_report(results, visual_path)

    for r in results:
        print(f"{r['market']}: {r['n_parent_candles']} candles, {r['n_entered']} entered, "
              f"efficiency p={r['efficiency']['p_value']}, direction p={r['direction_agreement']['p_value']}, "
              f"lag1 autocorr={r['lag1_autocorr_5m_returns']}")

    print(f"\nTechnical report: {technical_path.resolve()}")
    print(f"Visual report: {visual_path.resolve()}")


if __name__ == "__main__":
    main()
