"""One check cycle of the live paper-trading lab (Phase 2): fetches fresh
data for all 5 markets, runs src.live.paper_trading.run_check, persists
the updated state, and appends any closed trades to the running log.

Meant to be invoked once per 1h candle close by a scheduled agent — this
script itself has no scheduling logic, it just does one check "as of now".
Running it more or less often than hourly is harmless (check_market is
idempotent against re-checking the same state), just wasteful or laggy.

Also logs a per-piece diagnostic trace on every check (src.live.diagnostics
— which pieces of the strategy passed/failed and the final yes/no) and
regenerates each market's interactive HTML chart
(src.report.live_diagnostics_chart), same instrumentation as
scripts/run_live_check_volatility.py — purely observational, does not
change the entry/exit decision or the sizing (still 100% of equity per
trade here, unlike that other script). Added so BOTH live containers give
the same level of "why did/didn't it enter" detail, not just the newer one.

Usage:
    .venv/bin/python scripts/run_live_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.data.fetcher import fetch_ohlcv
from src.live.diagnostics import diagnose_entry
from src.live.paper_trading import default_state, run_check
from src.report.live_diagnostics_chart import write_live_diagnostics_chart
from src.report.paths import _slug
from src.strategies.builder import load_strategy

STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"

# Comfortably more than lookback_bars=200 (1h) / higher_tf_lookback_bars=50
# (1d) so a fetch never comes up short.
PARENT_SINCE_DAYS = 20
HIGHER_TF_SINCE_DAYS = 90

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_lab"
STATE_PATH = STATE_DIR / "state.json"
TRADE_LOG_PATH = STATE_DIR / "trades.csv"
EVENT_LOG_PATH = STATE_DIR / "events.log"
DIAGNOSTICS_DIR = STATE_DIR / "diagnostics"
CHARTS_DIR = STATE_DIR / "charts"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return default_state()


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _append_closed_trades(events: list[dict]) -> None:
    closed = [e for e in events if e["type"] == "closed"]
    if not closed:
        return
    rows = []
    for event in closed:
        row = {"market": event["market"], **event["trade"]}
        for account_key, account in event["accounts"].items():
            row[f"equity_before_{account_key}"] = account["equity_before"]
            row[f"equity_after_{account_key}"] = account["equity_after"]
        rows.append(row)
    new_rows = pd.DataFrame(rows)
    if TRADE_LOG_PATH.exists():
        existing = pd.read_csv(TRADE_LOG_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TRADE_LOG_PATH, index=False)


def _diagnostics_path(symbol: str) -> Path:
    return DIAGNOSTICS_DIR / f"{_slug(symbol)}.json"


def _load_diagnostics(symbol: str) -> dict:
    path = _diagnostics_path(symbol)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_diagnostics(symbol: str, diagnostics: dict) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    _diagnostics_path(symbol).write_text(json.dumps(diagnostics, indent=2))


def main() -> None:
    strategy = load_strategy(STRATEGY_PATH)
    state = _load_state()

    parent_since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=PARENT_SINCE_DAYS)).strftime("%Y-%m-%d")
    higher_tf_since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HIGHER_TF_SINCE_DAYS)).strftime("%Y-%m-%d")

    data_by_market = {}
    for market in MARKETS:
        df = fetch_ohlcv(market.symbol, TIMEFRAME, since=parent_since, use_cache=False)
        higher_tf_df = fetch_ohlcv(market.symbol, HIGHER_TIMEFRAME, since=higher_tf_since, use_cache=False)
        data_by_market[market.symbol] = (df, higher_tf_df)

    new_state, events = run_check(strategy, data_by_market, state)
    _save_state(new_state)
    _append_closed_trades(events)

    # Diagnostic trace + chart, for EVERY market, every check — independent
    # of run_check's own state machine (see src/live/diagnostics.py).
    for market in MARKETS:
        df, higher_tf_df = data_by_market[market.symbol]
        trace = diagnose_entry(df, higher_tf_df, strategy)
        diagnostics = _load_diagnostics(market.symbol)
        diagnostics[trace["timestamp"]] = trace
        _save_diagnostics(market.symbol, diagnostics)
        write_live_diagnostics_chart(
            df.tail(200), diagnostics, market.symbol, CHARTS_DIR / f"{_slug(market.symbol)}.html"
        )

    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    lines = [f"[{timestamp}] check complete — {len(events)} event(s)"]
    for event in events:
        lines.append(f"  {event['market']}: {event['type']}"
                      + (f" ({event['trade']['exit_reason']}, {event['trade']['pnl_pct']:.2f}%)" if event["type"] == "closed" else ""))
    if not events:
        lines.append("  (no changes — no open/pending positions and no new signals)")
    report = "\n".join(lines)

    print(report)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\naccounts: {json.dumps(new_state['accounts'], indent=2)}")
    print(f"state: {STATE_PATH}")
    print(f"trade log: {TRADE_LOG_PATH}")
    print(f"diagnostics charts: {CHARTS_DIR}/<market>.html")


if __name__ == "__main__":
    main()
