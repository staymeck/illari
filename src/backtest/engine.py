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

Stop goes below the support level, target is a multiple R of the risk.

A note on look-ahead bias: at every step `t`, the analysis functions are only
given a fixed-size trailing window `df.iloc[t + 1 - lookback_bars : t + 1]`
(nothing from the future, and never a growing window). Pivots from
`find_swing_points` use a centered rolling window, so the last `order` bars of
any slice are automatically left unconfirmed (NaN -> False) until enough
"future" bars exist within the slice itself — i.e. the function is already
look-ahead safe as long as it's only fed data up to `t`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.markets import session_for_hour
from src.analysis.candles import bullish_reversal_pattern
from src.analysis.fibonacci import is_near_confluence, latest_up_leg
from src.analysis.structure import classify_trend, support_resistance_levels


@dataclass(frozen=True)
class BacktestConfig:
    lookback_bars: int = 200          # trailing window used to compute structure
    swing_order: int = 3              # see structure.find_swing_points
    support_tolerance_pct: float = 1.0  # how close to support counts as a touch
    fib_tolerance_pct: float = 1.0    # how close to a Fibonacci level counts as confluence
    require_candle_confirmation: bool = True  # require a bullish reversal candle to confirm
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


def _find_entry_signal(history: pd.DataFrame, cfg: BacktestConfig) -> EntrySignal | None:
    """Returns the confluence signal that triggers an entry, or None if there's
    no signal, evaluating only with data up to the last row of `history` (no future)."""
    if len(history) < cfg.lookback_bars:
        return None

    trend = classify_trend(history, order=cfg.swing_order)
    if trend != "uptrend":
        return None

    levels = support_resistance_levels(history, order=cfg.swing_order)
    supports = levels["support"]
    if not supports:
        return None

    last_close = history["close"].iloc[-1]
    matched_support = None
    for level in supports:
        distance_pct = abs(last_close - level) / level * 100
        if last_close >= level and distance_pct <= cfg.support_tolerance_pct:
            matched_support = level
            break
    if matched_support is None:
        return None

    leg = latest_up_leg(history, order=cfg.swing_order)
    if leg is None:
        return None
    confluence = is_near_confluence(last_close, leg["low"], leg["high"], cfg.fib_tolerance_pct)
    if confluence is None:
        return None
    fib_ratio, fib_level = confluence

    pattern = bullish_reversal_pattern(history, idx=len(history) - 1)
    if cfg.require_candle_confirmation and pattern is None:
        return None

    return EntrySignal(support_level=matched_support, fib_ratio=fib_ratio, fib_level=fib_level, pattern=pattern)


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Simulates the strategy over `df` (columns: timestamp, open, high, low,
    close, volume) and returns (trades, equity_curve)."""
    cfg = cfg or BacktestConfig()
    df = df.reset_index(drop=True)

    trades: list[dict] = []
    equity = cfg.initial_equity
    equity_curve: list[float] = [equity]

    i = cfg.lookback_bars
    n = len(df)
    while i < n - 1:
        window_start = max(0, i + 1 - cfg.lookback_bars)
        history = df.iloc[window_start : i + 1]
        signal = _find_entry_signal(history, cfg)

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
