"""Event-driven backtest engine: simulates strategy signal execution over
the entry candles, candle by candle, without looking ahead.

Simplifications of this phase (documented, not hidden):
  - Only one open position at a time.
  - Within a single 5m candle, if its range touches both the SL and the
    TP, the SL is assumed to execute first (conservative/pessimistic
    assumption, so results aren't artificially inflated).
  - Entry executes at the close price of the candle that generated the
    signal (that's when all of that candle's information is already
    available), with slippage applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_lab.strategy.base import Signal


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stop_loss' | 'take_profit' | 'end_of_data'
    confidence_score: int
    hour_utc: int
    session: str
    reasons: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.DataFrame  # columns: timestamp, equity
    final_equity: float


def _apply_slippage(price: float, direction: str, side: str, slippage_pct: float) -> float:
    """`side`: 'entry' or 'exit'. Slippage always works against the trader."""
    factor = slippage_pct / 100.0
    if (direction == "long" and side == "entry") or (direction == "short" and side == "exit"):
        return price * (1 + factor)
    return price * (1 - factor)


def run_backtest(
    signals: list[Signal],
    entry_df: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_pct: float,
    commission_pct: float,
    slippage_pct: float,
) -> BacktestResult:
    df = entry_df.reset_index(drop=True)
    ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}
    signals_sorted = sorted(signals, key=lambda s: s.timestamp)

    capital = initial_capital
    trades: list[Trade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = [(df["timestamp"].iloc[0], capital)]

    open_until_idx = -1  # candle index up to which a position is open

    for sig in signals_sorted:
        entry_idx = ts_to_idx.get(sig.timestamp)
        if entry_idx is None or entry_idx <= open_until_idx:
            continue  # signal out of range, or a position is already open over it

        raw_entry = sig.entry_price
        entry_price = _apply_slippage(raw_entry, sig.direction, "entry", slippage_pct)

        risk_per_unit = abs(entry_price - sig.stop_loss)
        if risk_per_unit <= 0:
            continue
        risk_amount = capital * (risk_per_trade_pct / 100.0)
        position_size = risk_amount / risk_per_unit

        exit_price, exit_time, exit_reason = None, None, "end_of_data"
        for j in range(entry_idx + 1, len(df)):
            bar = df.iloc[j]
            hit_sl = (sig.direction == "long" and bar["low"] <= sig.stop_loss) or (
                sig.direction == "short" and bar["high"] >= sig.stop_loss
            )
            hit_tp = (sig.direction == "long" and bar["high"] >= sig.take_profit) or (
                sig.direction == "short" and bar["low"] <= sig.take_profit
            )
            if hit_sl:  # conservative assumption: SL before TP if both happen within the same candle
                exit_price, exit_time, exit_reason = sig.stop_loss, bar["timestamp"], "stop_loss"
                break
            if hit_tp:
                exit_price, exit_time, exit_reason = sig.take_profit, bar["timestamp"], "take_profit"
                break

        if exit_price is None:
            exit_price = df["close"].iloc[-1]
            exit_time = df["timestamp"].iloc[-1]
            j = len(df) - 1

        exit_price_slipped = _apply_slippage(exit_price, sig.direction, "exit", slippage_pct)

        gross_pnl = (
            (exit_price_slipped - entry_price) * position_size
            if sig.direction == "long"
            else (entry_price - exit_price_slipped) * position_size
        )
        notional_entry = entry_price * position_size
        notional_exit = exit_price_slipped * position_size
        commission = (notional_entry + notional_exit) * (commission_pct / 100.0)
        pnl = gross_pnl - commission
        pnl_pct = pnl / capital * 100.0

        capital += pnl
        open_until_idx = j

        trades.append(
            Trade(
                entry_time=sig.timestamp,
                exit_time=exit_time,
                direction=sig.direction,
                entry_price=entry_price,
                exit_price=exit_price_slipped,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                position_size=position_size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                confidence_score=sig.confidence_score,
                hour_utc=sig.hour_utc,
                session=sig.session,
                reasons=sig.reasons,
            )
        )
        equity_points.append((exit_time, capital))

    equity_curve = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    return BacktestResult(trades=trades, equity_curve=equity_curve, final_equity=capital)
