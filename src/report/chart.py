"""Interactive candlestick chart with entries marked by which approach
produced them (deterministic / probabilistic / random) — see docs/PLAN.md.
Uses plotly (already in requirements.txt), writes a standalone HTML file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.report.paths import REPORTS_DIR

_MARKERS = {
    "deterministic": {"color": "#2ca02c", "symbol": "triangle-up", "size": 12},
    "probabilistic": {"color": "#1f77b4", "symbol": "diamond", "size": 11},
    "random": {"color": "#7f7f7f", "symbol": "circle-open", "size": 9},
}


def render_entries_chart(
    df: pd.DataFrame, trades_by_kind: dict[str, pd.DataFrame], symbol: str, timeframe: str
) -> str:
    """Builds a candlestick chart of `df` with one marker trace per kind in
    `trades_by_kind` (expected keys: "deterministic", "probabilistic",
    "random" — any subset is fine, unknown keys just get a default marker),
    plotted at each trade's entry_time/entry_price. Returns the chart as a
    standalone HTML string."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol,
            )
        ]
    )

    for kind, trades in trades_by_kind.items():
        if trades is None or trades.empty:
            continue
        style = _MARKERS.get(kind, {"color": "#d62728", "symbol": "x", "size": 10})
        fig.add_trace(
            go.Scatter(
                x=trades["entry_time"],
                y=trades["entry_price"],
                mode="markers",
                name=f"{kind} entry",
                marker=dict(color=style["color"], symbol=style["symbol"], size=style["size"]),
            )
        )

    fig.update_layout(
        title=f"{symbol} · {timeframe} — entries by approach",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def _line_segments(pairs: pd.DataFrame) -> tuple[list, list]:
    """Builds x/y arrays for one Scatter trace drawing many disconnected
    entry->exit line segments (None between each pair breaks the line),
    instead of one trace per trade — keeps trace count small regardless of
    how many round trips are being drawn."""
    xs: list = []
    ys: list = []
    for _, row in pairs.iterrows():
        xs += [row["entry_time"], row["exit_time"], None]
        ys += [row["entry_price"], row["exit_price"], None]
    return xs, ys


def render_opportunity_chart(
    df: pd.DataFrame, real_trades: pd.DataFrame, ideal_trades: pd.DataFrame, symbol: str, timeframe: str
) -> str:
    """Candlestick chart contrasting what a real strategy actually traded
    against the oracle's ideal round trips (src/analysis/opportunity_scan.
    find_ideal_trades) — built to visually answer "what opportunity or
    noise are we missing", see docs/PLAN.md / Bitácora Illari.

    `real_trades`: entry_time/entry_price/exit_time/exit_price/pnl_pct/
    exit_reason (the engine's trade record). `ideal_trades`: direction/
    entry_time/entry_price/exit_time/exit_price/mfe_pct (find_ideal_trades'
    output). Either can be empty.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name=symbol,
            )
        ]
    )

    if real_trades is not None and not real_trades.empty:
        xs, ys = _line_segments(real_trades)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="#555555", width=1),
            name="real trade path", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=real_trades["entry_time"], y=real_trades["entry_price"], mode="markers",
            name="real entry", marker=dict(color="#2ca02c", symbol="triangle-up", size=11),
        ))
        fig.add_trace(go.Scatter(
            x=real_trades["exit_time"], y=real_trades["exit_price"], mode="markers",
            name="real exit", customdata=real_trades[["pnl_pct", "exit_reason"]],
            hovertemplate="exit %{x}<br>price %{y}<br>pnl %{customdata[0]:.2f}%<br>reason %{customdata[1]}<extra></extra>",
            marker=dict(color="#d62728", symbol="triangle-down", size=11),
        ))

    for direction, color, label in [("long", "#1f77b4", "ideal long"), ("short", "#ff7f0e", "ideal short")]:
        subset = (
            ideal_trades[ideal_trades["direction"] == direction]
            if ideal_trades is not None and not ideal_trades.empty
            else pd.DataFrame()
        )
        if subset.empty:
            continue
        xs, ys = _line_segments(subset)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=color, width=1, dash="dot"),
            name=f"{label} path", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=subset["entry_time"], y=subset["entry_price"], mode="markers",
            name=f"{label} entry", marker=dict(color=color, symbol="circle", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=subset["exit_time"], y=subset["exit_price"], mode="markers",
            name=f"{label} exit (best possible)", customdata=subset[["mfe_pct"]],
            hovertemplate="exit %{x}<br>price %{y}<br>MFE %{customdata[0]:.2f}%<extra></extra>",
            marker=dict(color=color, symbol="circle-open", size=9),
        ))

    fig.update_layout(
        title=f"{symbol} · {timeframe} — real trades vs. oracle opportunity",
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        legend=dict(orientation="h"),
        height=750,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def write_opportunity_chart(
    df: pd.DataFrame,
    real_trades: pd.DataFrame,
    ideal_trades: pd.DataFrame,
    symbol: str,
    timeframe: str,
    suffix: str = "opportunity_vs_reality",
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    safe_symbol = symbol.replace("/", "-").lower()
    out_path = reports_dir / f"{safe_symbol}_{timeframe}_{suffix}.html"
    html = render_opportunity_chart(df, real_trades, ideal_trades, symbol, timeframe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_entries_chart(
    df: pd.DataFrame,
    trades_by_kind: dict[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    safe_symbol = symbol.replace("/", "-").lower()
    out_path = reports_dir / f"{safe_symbol}_{timeframe}_entries_chart.html"
    html = render_entries_chart(df, trades_by_kind, symbol, timeframe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
