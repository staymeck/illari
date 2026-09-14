"""Per-trade audit trail: renders a trades DataFrame (from
src.backtest.engine.run_backtest) so every single trade can be inspected —
what called the trend, which pattern fired, and exactly why it exited when
it did — instead of only trusting the aggregate win rate. See docs/PLAN.md /
Bitácora Illari: motivated directly by wanting case-by-case traceability of
individual decisions, not just an average.
"""
from __future__ import annotations

import pandas as pd

# Only the columns actually present in a given strategy's trades survive —
# which extras exist depends entirely on which confirmations its YAML uses.
_AUDIT_COLUMNS = [
    "entry_time",
    "context_trend",
    "higher_tf_trend",
    "pattern",
    "fib_ratio",
    "volume_bias",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "stop_was_premature",
    "pnl_pct",
]


def build_trade_audit(trades: pd.DataFrame) -> pd.DataFrame:
    """Selects and orders the columns relevant to a trade-by-trade audit,
    chronologically. Only columns actually present in `trades` are kept."""
    if trades.empty:
        return trades
    columns = [c for c in _AUDIT_COLUMNS if c in trades.columns]
    return trades[columns].sort_values("entry_time").reset_index(drop=True)


def render_trade_audit_markdown(trades: pd.DataFrame, title: str) -> str:
    """One markdown table, one row per trade, in chronological order."""
    audit = build_trade_audit(trades)
    lines = [f"## {title}", ""]
    if audit.empty:
        lines.append("_no trades_")
        lines.append("")
        return "\n".join(lines)

    header = list(audit.columns)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, row in audit.iterrows():
        cells = []
        for col in header:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 4)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def pattern_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregates by candlestick pattern (if the strategy used one): count,
    win rate, average pnl, and exit-reason distribution — to see which
    specific pattern is carrying the result versus dragging it down,
    instead of one blended average across all of them."""
    if trades.empty or "pattern" not in trades.columns:
        return pd.DataFrame()

    rows = []
    for pattern, group in trades.groupby("pattern"):
        wins = group[group["pnl_abs"] > 0]
        rows.append(
            {
                "pattern": pattern,
                "n_trades": len(group),
                "win_rate_pct": round(100 * len(wins) / len(group), 2),
                "avg_pnl_pct": round(group["pnl_pct"].mean(), 3),
                "stop_pct": round(100 * (group["exit_reason"] == "stop").mean(), 1),
                "target_pct": round(100 * (group["exit_reason"] == "target").mean(), 1),
                "max_holding_pct": round(100 * (group["exit_reason"] == "max_holding_bars").mean(), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_pnl_pct", ascending=False).reset_index(drop=True)
