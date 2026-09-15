"""Phase 2 of the original project plan: a live paper-trading lab running
the SAME deterministic strategy already validated in the historical lab —
no AI in the decision loop (see the LLM-scoring experiments' negative
result, docs/PLAN.md / Bitácora Illari — this module never imports
src.analysis.llm_score). Persistent state across checks (unlike
evaluate_signal's single stateless snapshot) tracks, per market, whether a
position is open, mirrored across three parallel simulated accounts
($10,000/$100/$10) to observe fee/rounding effects at different capital
scales, per the user's original request.

Designed to be invoked once per check (e.g. hourly, at each 1h candle
close, by a scheduled agent) — this module has NO scheduling logic of its
own and does NOT fetch data itself: `run_check` takes already-fetched
OHLCV per market, so its core logic is pure and testable with synthetic
data, same convention as the rest of the engine.

Per-market position lifecycle, mirroring the backtest engine's own
"decide at close, execute at next bar's open" discipline exactly (see
src/backtest/engine.py) instead of approximating it:
  None -> "pending" (signal fired at bar i's close, execution price not
  known yet) -> "open" (bar i+1 has now closed, so its open price — the
  real entry — is known) -> None again once stop/target/timeout closes it.
Bars at or before the signal bar are immutable history, so re-deriving the
exact same ctx/signal one check later (via `_evaluate_at`) is always safe
and gives an identical result — no need to serialize a DataFrame into the
JSON state file, just a timestamp.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.structure import find_swing_points
from src.backtest.engine import (
    _build_trade_record,
    _confirmed_pivots_as_of,
    _find_entry_signal,
    _higher_tf_window_as_of,
    _walk_to_exit,
)
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext

ACCOUNT_SIZES = [10000.0, 100.0, 10.0]
MIN_NOTIONAL_USD = 5.0  # Binance spot's typical minimum order value - informational only, never blocks a trade here


def default_state() -> dict:
    return {"markets": {}, "accounts": {str(int(size)): size for size in ACCOUNT_SIZES}}


def _evaluate_at(
    df: pd.DataFrame,
    strategy: ResolvedStrategy,
    higher_tf_df: pd.DataFrame | None,
    at_index: int,
    higher_tf_lookback_bars: int = 50,
):
    """Rebuilds the exact ctx + entry-signal check the engine would have
    made at bar `at_index` when IT was the most recently closed bar — works
    even when `df` now has later rows too, since bars <= at_index are
    immutable history. Mirrors run_backtest/evaluate_signal's own window
    construction."""
    marked_full = find_swing_points(df, order=strategy.swing_order)
    window_start = max(0, at_index + 1 - strategy.lookback_bars)
    marked_window = _confirmed_pivots_as_of(marked_full, at_index, window_start, strategy.swing_order)
    price_window = df.iloc[window_start : at_index + 1]

    higher_tf_window = None
    if higher_tf_df is not None:
        interval = higher_tf_df["timestamp"].diff().median()
        higher_tf_window = _higher_tf_window_as_of(
            higher_tf_df, df["timestamp"].iloc[at_index], higher_tf_lookback_bars, interval
        )

    ctx = EvalContext(price_window=price_window, marked_window=marked_window, higher_tf_window=higher_tf_window)
    return ctx, _find_entry_signal(ctx, strategy)


def check_market(
    df: pd.DataFrame,
    higher_tf_df: pd.DataFrame | None,
    strategy: ResolvedStrategy,
    market_state: dict | None,
    size_multiplier_fn=None,
) -> tuple[dict | None, list[dict]]:
    """One check for one market: given fresh OHLCV and the market's current
    state (None / "pending" / "open"), returns (new_state, events) —
    events are plain dicts for logging, never raise on their own.

    `size_multiplier_fn` (optional): `EvalContext -> float`, called once at
    the moment a pending signal resolves into an open position, to decide
    what fraction of equity that trade risks (see src/live/position_sizing.py
    — e.g. volatility_size_multiplier). The chosen fraction is frozen into
    `market_state["size_multiplier"]` for that trade's whole lifetime (a
    later check must never re-price the size of an already-open trade).
    Omitted (the default): every trade risks 100% of equity, identical to
    this module's behavior before this parameter existed."""
    size_multiplier_fn = size_multiplier_fn or (lambda ctx: 1.0)
    df = df.reset_index(drop=True)
    events: list[dict] = []
    status = market_state.get("status") if market_state else None

    if status == "open":
        entry_time = pd.Timestamp(market_state["entry_time"])
        matches = df.index[df["timestamp"] == entry_time]
        if len(matches) == 0:
            events.append({"type": "error", "detail": "entry bar rolled out of the fetched window"})
            return market_state, events
        entry_idx = int(matches[0])
        bars_elapsed = (len(df) - 1) - entry_idx

        exit_idx, exit_price, exit_reason, stop_was_premature = _walk_to_exit(
            df, entry_idx, market_state["stop_price"], market_state["target_price"],
            strategy.stop_trigger, strategy.max_holding_bars,
        )
        # _walk_to_exit reports "timeout" whenever it runs out of AVAILABLE
        # bars to scan, whether or not max_holding_bars has genuinely
        # elapsed yet — with live data that's usually just "not enough
        # history fetched yet, still open", not a real timeout.
        genuinely_done = exit_reason in ("stop", "target") or bars_elapsed >= strategy.max_holding_bars
        if not genuinely_done:
            return market_state, events

        trade, _ = _build_trade_record(
            df, entry_idx, exit_idx, market_state["entry_price"], exit_price, exit_reason,
            stop_was_premature, market_state["stop_price"], market_state["target_price"],
            strategy.fee_pct, 1.0, market_state.get("extras", {}),
        )
        trade["entry_time"] = str(trade["entry_time"])
        trade["exit_time"] = str(trade["exit_time"])
        events.append({"type": "closed", "trade": trade, "size_multiplier": market_state.get("size_multiplier", 1.0)})
        return None, events

    if status == "pending":
        signal_bar_time = pd.Timestamp(market_state["signal_bar_time"])
        matches = df.index[df["timestamp"] == signal_bar_time]
        if len(matches) == 0:
            events.append({"type": "error", "detail": "signal bar rolled out of the fetched window"})
            return None, events
        signal_idx = int(matches[0])
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            return market_state, events  # the execution bar hasn't closed yet

        ctx, signal = _evaluate_at(df, strategy, higher_tf_df, signal_idx)
        if signal is None:
            # Bars <= signal_idx are immutable, so this shouldn't normally
            # happen — fails safe by dropping the pending signal rather
            # than opening a position nothing actually confirmed.
            events.append({"type": "signal_vanished"})
            return None, events

        setup, extras = signal
        entry_price = float(df["open"].iloc[entry_idx])
        stop_price = strategy.stop_fn(entry_price, setup, ctx, strategy.stop_params)
        target_price = strategy.target_fn(entry_price, stop_price, ctx, strategy.target_params)
        # Sized once, here, using the same ctx (as of the signal bar's
        # close) the entry decision itself used — never re-priced later.
        size_multiplier = float(size_multiplier_fn(ctx))
        new_state = {
            "status": "open",
            "entry_time": str(df["timestamp"].iloc[entry_idx]),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "extras": extras,
            "size_multiplier": size_multiplier,
        }
        events.append({
            "type": "opened", "entry_price": entry_price, "stop_price": stop_price,
            "target_price": target_price, "size_multiplier": size_multiplier,
        })
        return new_state, events

    # status is None: look for a fresh signal on the latest closed bar.
    at_index = len(df) - 1
    if at_index < strategy.lookback_bars:
        return None, events
    _, signal = _evaluate_at(df, strategy, higher_tf_df, at_index)
    if signal is None:
        return None, events
    new_state = {"status": "pending", "signal_bar_time": str(df["timestamp"].iloc[at_index])}
    events.append({"type": "signal_pending"})
    return new_state, events


def run_check(
    strategy: ResolvedStrategy,
    data_by_market: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]],
    state: dict | None,
    size_multiplier_fn=None,
) -> tuple[dict, list[dict]]:
    """Runs check_market for every market in `data_by_market`, updates each
    of the 3 parallel accounts on every closed trade (100% of that
    account's current equity per trade, scaled by `size_multiplier_fn`
    when given — see check_market's own docstring; omitted, this is
    exactly the 100%-of-equity convention run_backtest itself uses), and
    flags (never blocks) when the actual sized notional would fall under
    Binance's typical minimum order value for the smallest account.
    Returns (new_state, events) — the caller owns persisting `new_state`
    and logging `events`."""
    state = state or default_state()
    all_events: list[dict] = []

    for symbol, (df, higher_tf_df) in data_by_market.items():
        market_state = state["markets"].get(symbol)
        new_market_state, events = check_market(df, higher_tf_df, strategy, market_state, size_multiplier_fn)
        state["markets"][symbol] = new_market_state

        for event in events:
            event["market"] = symbol
            if event["type"] == "opened":
                size_multiplier = event.get("size_multiplier", 1.0)
                event["accounts"] = {}
                for account_key, equity in state["accounts"].items():
                    notional_usd = equity * size_multiplier
                    event["accounts"][account_key] = {
                        "notional_usd": round(notional_usd, 2),
                        "below_min_notional": notional_usd < MIN_NOTIONAL_USD,
                    }
            elif event["type"] == "closed":
                pnl_pct = event["trade"]["pnl_pct"]
                size_multiplier = event.get("size_multiplier", 1.0)
                event["accounts"] = {}
                for account_key in list(state["accounts"]):
                    equity_before = state["accounts"][account_key]
                    equity_after = equity_before * (1 + size_multiplier * pnl_pct / 100)
                    event["accounts"][account_key] = {
                        "equity_before": round(equity_before, 2),
                        "equity_after": round(equity_after, 2),
                    }
                    state["accounts"][account_key] = equity_after

        all_events.extend(events)

    return state, all_events
