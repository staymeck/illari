"""Generic backtest engine: runs a config-driven strategy (src.strategies)
composed of a context piece, a setup piece, a list of confirmation pieces,
and stop/target risk pieces.

See docs/PLAN.md, "Config-driven strategy catalog". Long-only in this
version (entries always buy, never short) — but the required market regime
IS configurable via `context.required` in the YAML (default "uptrend",
matching the one hardcoded strategy the engine ran before this
generalization — see config/strategies/trend_pullback_fib.yaml, which
reproduces it exactly for regression parity). A mean-reversion strategy, for
instance, sets `required: sideways` so it isn't gated behind an uptrend that
range-bound setups don't need; `required: any` skips the context check
entirely.

Performance note: find_swing_points is computed exactly once for the whole
series (vectorized, O(n)) instead of being recomputed from scratch on a
trailing slice at every single bar — this used to be the dominant cost of a
run (see docs/PLAN.md discussion on 5 markets x 3 timeframes x 36 months).
Pieces that don't need pivots (e.g. future moving-average-based ones) just
ignore `marked_window`; precomputing it unconditionally is cheap and keeps
the engine simple.

A note on look-ahead bias: a swing pivot at bar `j` needs `swing_order` bars
*after* it to be confirmed (see structure.find_swing_points), so it's only
"knowable" from bar `j + swing_order` onward. Even though pivots are
precomputed for the entire series upfront, at every step `i` we only ever
look at pivots with `j + swing_order <= i` (via `_confirmed_pivots_as_of`) —
so the backtest never uses a pivot before it would really have been
confirmed.

Stop trigger (`risk.stop.trigger` in the YAML, "intrabar" by default) and the
`stop_was_premature` diagnostic are both directly informed by Kaufman,
*Trading Systems and Methods* (5th ed.), ch. 23 "Risk Control":
- "intrabar" (default, matches the original hardcoded strategy exactly):
  exits the instant the bar's low touches the stop — realistic (a resting
  stop order fires on touch), but exposed to a single noisy wick.
- "close": only confirms the stop if the bar's *close* is through it
  ("...stop-loss orders are usually based on the closing price... to take
  advantage of a pullback" — Kaufman). Targets are deliberately NOT given
  the same option: Kaufman recommends the opposite asymmetry for
  profit-taking ("you would want to exit at the time of the intraday spike
  rather than waiting for the end of the day"), so targets always trigger
  intrabar here.
- `stop_was_premature` records, for every stop-exited trade, whether price
  would have reached the target anyway before the holding window ran out
  had the stop not fired — a direct measurement of how much of the stop-out
  count is "noise" versus a real trend failure (see
  src/report/render.py's breakdown).

Optional multi-timeframe confirmation (Kaufman, ch. 19 "Multiple Time
Frames" — Elder's Triple Screen, Krausz, Pring's KST all use this same core
idea: don't take a lower-timeframe signal against the higher-timeframe
trend): pass `higher_tf_df` to `run_backtest` and add the
`higher_tf_trend` piece to a strategy's confirmations. Omitting
`higher_tf_df` (the default) leaves this feature fully inert — identical
behavior to every run before it existed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.markets import session_for_hour
from src.analysis.structure import find_swing_points
from src.strategies.builder import ResolvedStrategy
from src.strategies.types import EvalContext


def _confirmed_pivots_as_of(marked_full: pd.DataFrame, i: int, window_start: int, swing_order: int) -> pd.DataFrame:
    """Slice of the precomputed pivots restricted to `[window_start, i]` AND
    to only those pivots already confirmable at bar `i` (see module
    docstring, "A note on look-ahead bias"). This is a cheap positional
    slice — no recomputation — which is what makes precomputing pivots once
    safe and fast."""
    confirmed_end = i - swing_order + 1  # exclusive
    if confirmed_end <= window_start:
        return marked_full.iloc[0:0]
    return marked_full.iloc[window_start:confirmed_end]


def _find_entry_signal(ctx: EvalContext, strategy: ResolvedStrategy) -> tuple | None:
    """Runs the strategy's context -> setup -> confirmations chain. Returns
    `(setup, extras)` on a full match, or None. `extras` merges every piece's
    reporting extras (e.g. support_level, fib_ratio, pattern, volume_bias) —
    which keys appear depends entirely on which pieces the config includes.
    Always includes `context_trend`: the context layer's own determination
    (e.g. "uptrend") — previously computed only to gate entry and then
    discarded, so a trade record couldn't say what called the trend at the
    time (see docs/PLAN.md / Bitácora Illari, trade-audit discussion)."""
    trend = strategy.context_fn(ctx, strategy.context_params)
    if strategy.required_context != "any" and trend != strategy.required_context:
        return None

    setup = strategy.setup_fn(ctx, strategy.setup_params)
    if setup is None:
        return None

    extras = dict(setup.extras)
    extras["context_trend"] = trend
    for confirmation in strategy.confirmations:
        result = confirmation.fn(ctx, confirmation.params)
        if result is None:
            return None
        extras.update(result.extras)

    return setup, extras


@dataclass(frozen=True)
class SignalEvaluation:
    """The result of checking a strategy's entry condition on the single
    most recent bar of a (small, recent) OHLCV window — no simulation loop,
    no trades, no execution. This is what a live "market snapshot" needs
    (see docs/PLAN.md, Phase 2): "does the strategy's entry condition hold
    right now?", using only a rolling window of recent candles — the same
    `lookback_bars` a live system would keep — instead of a full historical
    backtest."""

    as_of: pd.Timestamp
    trend: str
    signal: bool
    reference_level: float | None
    extras: dict


def evaluate_signal(
    df: pd.DataFrame,
    strategy: ResolvedStrategy,
    higher_tf_df: pd.DataFrame | None = None,
    higher_tf_lookback_bars: int = 50,
) -> SignalEvaluation:
    """Evaluates `strategy`'s context -> setup -> confirmations chain on the
    LAST bar of `df` only. `df` only needs to cover `strategy.lookback_bars`
    (+ a small buffer for swing confirmation) — not a multi-year history —
    matching how little data this actually requires in practice (see
    docs/PLAN.md's discussion of how far back a live system needs to look).
    Raises ValueError if `df` is too short to evaluate at all.
    """
    df = df.reset_index(drop=True)
    i = len(df) - 1
    if i < strategy.lookback_bars:
        raise ValueError(f"evaluate_signal needs at least {strategy.lookback_bars} bars, got {i + 1}")

    window_start = max(0, i + 1 - strategy.lookback_bars)
    price_window = df.iloc[window_start : i + 1]
    marked_full = find_swing_points(df, order=strategy.swing_order)
    marked_window = _confirmed_pivots_as_of(marked_full, i, window_start, strategy.swing_order)

    higher_tf_window = None
    if higher_tf_df is not None:
        higher_tf_df = higher_tf_df.reset_index(drop=True)
        interval = higher_tf_df["timestamp"].diff().median()
        higher_tf_window = _higher_tf_window_as_of(
            higher_tf_df, df["timestamp"].iloc[i], higher_tf_lookback_bars, interval
        )

    ctx = EvalContext(price_window=price_window, marked_window=marked_window, higher_tf_window=higher_tf_window)
    trend = strategy.context_fn(ctx, strategy.context_params)
    signal = _find_entry_signal(ctx, strategy)

    as_of = df["timestamp"].iloc[i]
    if signal is None:
        return SignalEvaluation(as_of=as_of, trend=trend, signal=False, reference_level=None, extras={})
    setup, extras = signal
    return SignalEvaluation(as_of=as_of, trend=trend, signal=True, reference_level=setup.reference_level, extras=extras)


def _walk_to_exit(
    df: pd.DataFrame,
    entry_idx: int,
    stop_price: float,
    target_price: float,
    stop_trigger: str,
    max_holding_bars: int,
) -> tuple[int, float, str, bool | None]:
    """Scans forward from `entry_idx` for a stop/target/timeout exit — the
    same walk used by both `run_backtest` and
    `src.backtest.random_benchmark` (which differs only in how it picks
    *when* to enter, not in how a position is managed once open). Returns
    `(exit_idx, exit_price, exit_reason, stop_was_premature)`."""
    n = len(df)
    last_possible = min(entry_idx + max_holding_bars, n - 1)
    for j in range(entry_idx, last_possible + 1):
        bar = df.iloc[j]
        stop_hit = bar["close"] <= stop_price if stop_trigger == "close" else bar["low"] <= stop_price
        if stop_hit:
            exit_price = bar["close"] if stop_trigger == "close" else stop_price
            # Diagnostic (per Kaufman, Trading Systems and Methods, ch. 23:
            # stops are "a duel with price noise" — a stop can capture the
            # worst of a move that reverses right after): would the target
            # still have been reached later, had this stop not fired?
            stop_was_premature = bool((df["high"].iloc[j + 1 : last_possible + 1] >= target_price).any())
            return j, exit_price, "stop", stop_was_premature
        if bar["high"] >= target_price:
            return j, target_price, "target", None
    exit_idx = last_possible
    return exit_idx, df["close"].iloc[exit_idx], "timeout", None


def _build_trade_record(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    stop_was_premature: bool | None,
    stop_price: float,
    target_price: float,
    fee_pct: float,
    equity_before: float,
    extras: dict,
) -> tuple[dict, float]:
    """Builds the trade dict (with fee-adjusted PnL, hour/session labels,
    and any piece extras merged in) and the resulting equity — shared by
    `run_backtest` and `random_benchmark.run_random_trial`. Returns
    `(trade_dict, new_equity)`."""
    gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
    net_pnl_pct = gross_pnl_pct - 2 * fee_pct  # entry + exit fees
    equity = equity_before * (1 + net_pnl_pct / 100)
    pnl_abs = equity - equity_before

    entry_ts = df["timestamp"].iloc[entry_idx]
    trade = {
        "entry_time": entry_ts,
        "exit_time": df["timestamp"].iloc[exit_idx],
        "hour_utc": entry_ts.hour,
        "session": session_for_hour(entry_ts.hour),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "stop_was_premature": stop_was_premature,
        "pnl_pct": net_pnl_pct,
        "pnl_abs": pnl_abs,
        "equity_after": equity,
    }
    trade.update(extras)
    return trade, equity


def _higher_tf_window_as_of(
    higher_tf_df: pd.DataFrame, timestamp: pd.Timestamp, lookback_bars: int, interval: pd.Timedelta
) -> pd.DataFrame:
    """Trailing window of `higher_tf_df` containing only candles already
    *closed* as of `timestamp` — a higher-timeframe candle starting at time
    `s` isn't knowable until `s + interval`, so it's excluded until then.
    Used for multi-timeframe confirmation (see EvalContext.higher_tf_window)."""
    cutoff = timestamp - interval
    idx = higher_tf_df["timestamp"].searchsorted(cutoff, side="right") - 1
    if idx < 0:
        return higher_tf_df.iloc[0:0]
    start = max(0, idx + 1 - lookback_bars)
    return higher_tf_df.iloc[start : idx + 1]


def run_backtest(
    df: pd.DataFrame,
    strategy: ResolvedStrategy,
    higher_tf_df: pd.DataFrame | None = None,
    higher_tf_lookback_bars: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulates `strategy` over `df` (columns: timestamp, open, high, low,
    close, volume) and returns (trades, equity_curve).

    `higher_tf_df` (optional): a second, higher-timeframe OHLCV series (e.g.
    1d candles while `df` is 1h) — when given, every bar's EvalContext also
    gets `higher_tf_window`, a trailing window of already-closed
    higher-timeframe candles, for pieces like
    confirmations/higher_tf_trend.py. Omitting it (the default) leaves
    `higher_tf_window` as None everywhere, identical to every strategy run
    before this parameter existed.
    """
    df = df.reset_index(drop=True)
    marked_full = find_swing_points(df, order=strategy.swing_order)  # computed once for the whole series

    higher_tf_interval = None
    if higher_tf_df is not None:
        higher_tf_df = higher_tf_df.reset_index(drop=True)
        higher_tf_interval = higher_tf_df["timestamp"].diff().median()

    trades: list[dict] = []
    equity = strategy.initial_equity
    equity_curve: list[float] = [equity]

    i = strategy.lookback_bars
    n = len(df)
    while i < n - 1:
        window_start = max(0, i + 1 - strategy.lookback_bars)
        price_window = df.iloc[window_start : i + 1]

        if len(price_window) < strategy.lookback_bars:
            equity_curve.append(equity)
            i += 1
            continue

        marked_window = _confirmed_pivots_as_of(marked_full, i, window_start, strategy.swing_order)
        higher_tf_window = None
        if higher_tf_df is not None:
            higher_tf_window = _higher_tf_window_as_of(
                higher_tf_df, df["timestamp"].iloc[i], higher_tf_lookback_bars, higher_tf_interval
            )
        ctx = EvalContext(price_window=price_window, marked_window=marked_window, higher_tf_window=higher_tf_window)
        signal = _find_entry_signal(ctx, strategy)

        if signal is None:
            equity_curve.append(equity)
            i += 1
            continue
        setup, extras = signal

        entry_idx = i + 1  # enter at the next candle's open (no look-ahead)
        if entry_idx >= n:
            break
        entry_price = df["open"].iloc[entry_idx]
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
            stop_price, target_price, strategy.fee_pct, equity, extras,
        )
        trades.append(trade)
        equity_curve.append(equity)

        i = exit_idx + 1  # don't overlap trades

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, name="equity")
    return trades_df, equity_series


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series, initial_equity: float) -> dict:
    """Standard lab metrics, same format as resources/report.md."""
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
            "final_equity": initial_equity,
        }

    wins = trades[trades["pnl_abs"] > 0]
    losses = trades[trades["pnl_abs"] <= 0]
    gross_win = wins["pnl_abs"].sum()
    gross_loss = -losses["pnl_abs"].sum()

    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    max_drawdown_pct = drawdown_pct.min()

    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100

    return {
        "n_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy": round(trades["pnl_abs"].mean(), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
    }
