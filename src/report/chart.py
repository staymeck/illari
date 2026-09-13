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
