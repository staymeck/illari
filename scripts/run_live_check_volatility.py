"""Second live paper-trading lab: SAME strategy/signals as
scripts/run_live_check.py (trend_pullback_htf_confluence — the entry/exit
rules are completely unchanged), but position size is scaled by the
current volatility regime instead of a fixed 100% of equity per trade —
see src/live/position_sizing.py for the idea and
scripts/run_volatility_sizing_backtest.py /
scripts/run_volatility_sizing_sweep.py for why (28/28 tested
(atr_period, lookback, min_mult) combinations beat uniform sizing on BOTH
avg return and avg drawdown against the frozen baseline sequence).
Parameters below are the sweep's "best avg_drawdown" pick, not "best avg
return" — the more conservative choice, deliberately avoiding picking the
single number that happened to score highest in the grid.

Also logs, for every market on every check, a full diagnostic trace of
which pieces of the strategy passed or failed on that bar (see
src/live/diagnostics.py) — purely observational, never influences the
actual entry/exit decision — and regenerates each market's interactive
HTML chart (src/report/live_diagnostics_chart.py) so the local file always
reflects the latest state.

Own state directory (data/live_lab_volatility/), separate from
scripts/run_live_check.py's (data/live_lab/) — this runs in its own
Docker service, in parallel with the original, so the two approaches can
be compared over time on genuinely fresh, identical market data.

Usage:
    .venv/bin/python scripts/run_live_check_volatility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.markets import MARKETS
from src.analysis.regime import volatility_percentile
from src.data.fetcher import fetch_ohlcv
from src.live.diagnostics import diagnose_entry
from src.live.paper_trading import default_state, run_check
from src.live.position_sizing import volatility_size_multiplier
from src.report.live_diagnostics_chart import write_live_diagnostics_chart
from src.report.paths import _slug
from src.strategies.builder import load_strategy

STRATEGY_PATH = "config/strategies/trend_pullback_htf_confluence.yaml"
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "1d"

PARENT_SINCE_DAYS = 20
HIGHER_TF_SINCE_DAYS = 90

# Volatility sizing parameters — the sweep's best-avg-drawdown pick (see
# module docstring): avg_return=1.88%, avg_drawdown=3.76% vs uniform's
# 0.31%/7.15%, across all 5 markets x known/virgin windows.
VOL_ATR_PERIOD = 14
VOL_LOOKBACK = 100
VOL_MIN_MULT = 0.10
VOL_MAX_MULT = 1.0

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_lab_volatility"
STATE_PATH = STATE_DIR / "state.json"
TRADE_LOG_PATH = STATE_DIR / "trades.csv"
EVENT_LOG_PATH = STATE_DIR / "events.log"
DIAGNOSTICS_DIR = STATE_DIR / "diagnostics"
CHARTS_DIR = STATE_DIR / "charts"


def _size_multiplier_fn(ctx) -> float:
    pct = volatility_percentile(ctx.price_window, atr_period=VOL_ATR_PERIOD, lookback=VOL_LOOKBACK)
    return volatility_size_multiplier(pct, min_mult=VOL_MIN_MULT, max_mult=VOL_MAX_MULT)


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
        row = {"market": event["market"], "size_multiplier": event.get("size_multiplier", 1.0), **event["trade"]}
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

    new_state, events = run_check(strategy, data_by_market, state, size_multiplier_fn=_size_multiplier_fn)
    _save_state(new_state)
    _append_closed_trades(events)

    # Diagnostic trace + chart, for EVERY market, every check — independent
    # of check_market's own state machine (see src/live/diagnostics.py).
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
        detail = f" ({event['trade']['exit_reason']}, {event['trade']['pnl_pct']:.2f}%, size={event.get('size_multiplier', 1.0):.2f}x)" \
            if event["type"] == "closed" else (f" (size={event.get('size_multiplier', 1.0):.2f}x)" if event["type"] == "opened" else "")
        lines.append(f"  {event['market']}: {event['type']}{detail}")
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
