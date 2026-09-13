"""Minimal backtest engine, used to validate the full pipeline (data -> signal
-> simulation -> metrics) before scaling up to 5 markets x 3 timeframes with
every component of the analysis engine.

Strategy used for this validation (see docs/PLAN.md "Technical analysis engine
components" — structure and Fibonacci as *confluence* filters, never
standalone signals, and candlestick patterns only combined with context):
buy when all of the following line up on the same candle —

1. The (Dow) trend is an uptrend.
2. Price is touching a confirmed support level.
3. Price also sits near a Fibonacci retracement level of the latest up-leg
   (independent confluence check, per docs/PLAN.md).
4. The candle itself shows a bullish reversal pattern (hammer / bullish
   engulfing) — this is the "context" that makes the pattern meaningful.
5. Volume confirms real buyer participation (above-average volume on a
   buyer-dominant trailing window) — without this, items 1-4 could just be
   a low-liquidity wiggle with no real interest behind it.

Stop goes below the support level, target is a multiple R of the risk.

Performance note: `find_swing_points` is computed exactly once for the whole
series (vectorized, O(n)) instead of being recomputed from scratch on a
trailing slice at every single bar — this used to be the dominant cost of a
run (see docs/PLAN.md discussion on 5 markets x 3 timeframes x 36 months).

A note on look-ahead bias: a swing pivot at bar `j` needs `swing_order` bars
*after* it to be confirmed (see structure.find_swing_points), so it's only
"knowable" from bar `j + swing_order` onward. Even though pivots are
precomputed for the entire series upfront, at every step `i` we only ever
look at pivots with `j + swing_order <= i` (via `_confirmed_pivots_as_of`) —
so the backtest never uses a pivot before it would really have been
confirmed. This is equivalent to (and, unlike a naive fixed-size resliced
window, doesn't spuriously hide pivots near the start of the lookback
window) the original approach of feeding a fresh, growing-then-trimmed slice
to `find_swing_points` at every bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.markets import session_for_hour
from src.analysis.candles import bullish_reversal_pattern
from src.analysis.fibonacci import is_near_confluence, latest_up_leg
from src.analysis.structure import classify_trend, find_swing_points, support_resistance_levels
from src.analysis.volume import confirms_buyer_pressure, directional_volume_bias


@dataclass(frozen=True)
class BacktestConfig:
    lookback_bars: int = 200          # trailing window used to compute structure
    swing_order: int = 3              # see structure.find_swing_points
    support_tolerance_pct: float = 1.0  # how close to support counts as a touch
    fib_tolerance_pct: float = 1.0    # how close to a Fibonacci level counts as confluence
    require_candle_confirmation: bool = True  # require a bullish reversal candle to confirm
    require_volume_confirmation: bool = True  # require buyer-dominant, above-average volume
    volume_window: int = 20
    volume_spike_threshold: float = 1.2
    volume_min_bias: float = 0.1
    stop_pct_below_support: float = 0.5  # stop = support * (1 - this%)
    reward_risk_ratio: float = 2.0    # target = entry + R * risk
    max_holding_bars: int = 24        # force-close if stop/target isn't hit before this
    fee_pct: float = 0.1              # Binance fee per side (~0.1% spot)
    initial_equity: float = 10_000.0


@dataclass(frozen=True)
class EntrySignal:
    support_level: float
    fib_ratio: float
    fib_level: float
    pattern: str | None
    volume_bias: float


def _confirmed_pivots_as_of(marked_full: pd.DataFrame, i: int, window_start: int, swing_order: int) -> pd.DataFrame:
    """Slice of the precomputed pivots restricted to `[window_start, i]` AND
    to only those pivots already confirmable at bar `i` (see module docstring,
    "A note on look-ahead bias"). This is a cheap positional slice — no
    recomputation — which is what makes precomputing pivots once safe and fast."""
    confirmed_end = i - swing_order + 1  # exclusive
    if confirmed_end <= window_start:
        return marked_full.iloc[0:0]
    return marked_full.iloc[window_start:confirmed_end]


def _find_entry_signal(
    price_window: pd.DataFrame, marked_window: pd.DataFrame, cfg: BacktestConfig
) -> EntrySignal | None:
    """Returns the confluence signal that triggers an entry, or None if
    there's no signal.

    `price_window` is the trailing OHLCV window up to and including the
    current bar (its own close/volume are known once it closes). `marked_window`
    is the subset of precomputed pivots that are both within the lookback
    window and already confirmable as of the current bar (see
    `_confirmed_pivots_as_of`) — structure/Fibonacci only ever see pivots they
    would really have known about at this point in time.
    """
    if len(price_window) < cfg.lookback_bars:
        return None

    trend = classify_trend(marked_window, order=cfg.swing_order, marked=marked_window)
    if trend != "uptrend":
        return None

    levels = support_resistance_levels(marked_window, order=cfg.swing_order, marked=marked_window)
    supports = levels["support"]
    if not supports:
        return None

    last_close = price_window["close"].iloc[-1]
    matched_support = None
    for level in supports:
        distance_pct = abs(last_close - level) / level * 100
        if last_close >= level and distance_pct <= cfg.support_tolerance_pct:
            matched_support = level
            break
    if matched_support is None:
        return None

    leg = latest_up_leg(marked_window, order=cfg.swing_order, marked=marked_window)
    if leg is None:
        return None
    confluence = is_near_confluence(last_close, leg["low"], leg["high"], cfg.fib_tolerance_pct)
    if confluence is None:
        return None
    fib_ratio, fib_level = confluence

    last_idx = len(price_window) - 1
    pattern = bullish_reversal_pattern(price_window, idx=last_idx)
    if cfg.require_candle_confirmation and pattern is None:
        return None

    if cfg.require_volume_confirmation and not confirms_buyer_pressure(
        price_window,
        idx=last_idx,
        window=cfg.volume_window,
        spike_threshold=cfg.volume_spike_threshold,
        min_bias=cfg.volume_min_bias,
    ):
        return None
    volume_bias = directional_volume_bias(price_window, window=cfg.volume_window)

    return EntrySignal(
        support_level=matched_support,
        fib_ratio=fib_ratio,
        fib_level=fib_level,
        pattern=pattern,
        volume_bias=volume_bias,
    )


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Simulates the strategy over `df` (columns: timestamp, open, high, low,
    close, volume) and returns (trades, equity_curve)."""
    cfg = cfg or BacktestConfig()
    df = df.reset_index(drop=True)
    marked_full = find_swing_points(df, order=cfg.swing_order)  # computed once for the whole series

    trades: list[dict] = []
    equity = cfg.initial_equity
    equity_curve: list[float] = [equity]

    i = cfg.lookback_bars
    n = len(df)
    while i < n - 1:
        window_start = max(0, i + 1 - cfg.lookback_bars)
        price_window = df.iloc[window_start : i + 1]
        marked_window = _confirmed_pivots_as_of(marked_full, i, window_start, cfg.swing_order)
        signal = _find_entry_signal(price_window, marked_window, cfg)

        if signal is None:
            equity_curve.append(equity)
            i += 1
            continue

        entry_idx = i + 1  # enter at the next candle's open (no look-ahead)
        if entry_idx >= n:
            break
        entry_price = df["open"].iloc[entry_idx]
        stop_price = signal.support_level * (1 - cfg.stop_pct_below_support / 100)
        risk = entry_price - stop_price
        if risk <= 0:
            equity_curve.append(equity)
            i += 1
            continue
        target_price = entry_price + cfg.reward_risk_ratio * risk

        exit_idx = None
        exit_price = None
        exit_reason = "timeout"
        last_possible = min(entry_idx + cfg.max_holding_bars, n - 1)
        for j in range(entry_idx, last_possible + 1):
            bar = df.iloc[j]
            if bar["low"] <= stop_price:
                exit_idx, exit_price, exit_reason = j, stop_price, "stop"
                break
            if bar["high"] >= target_price:
                exit_idx, exit_price, exit_reason = j, target_price, "target"
                break
        if exit_idx is None:
            exit_idx = last_possible
            exit_price = df["close"].iloc[exit_idx]

        gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
        net_pnl_pct = gross_pnl_pct - 2 * cfg.fee_pct  # entry + exit fees
        trade_equity_before = equity
        equity *= 1 + net_pnl_pct / 100
        pnl_abs = equity - trade_equity_before

        entry_ts = df["timestamp"].iloc[entry_idx]
        trades.append(
            {
                "entry_time": entry_ts,
                "exit_time": df["timestamp"].iloc[exit_idx],
                "hour_utc": entry_ts.hour,
                "session": session_for_hour(entry_ts.hour),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "fib_ratio": signal.fib_ratio,
                "pattern": signal.pattern,
                "volume_bias": signal.volume_bias,
                "pnl_pct": net_pnl_pct,
                "pnl_abs": pnl_abs,
                "equity_after": equity,
            }
        )
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
