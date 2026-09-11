"""Interactive candlestick chart (Plotly) per scenario: candlestick +
structure + Fibonacci zone + trade entries/exits + volume + session
shading. Exported as a standalone HTML file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SESSION_COLORS = {
    "asia": "rgba(99, 110, 250, 0.08)",
    "london": "rgba(0, 204, 150, 0.08)",
    "new_york": "rgba(239, 85, 59, 0.08)",
    "overlap_london_ny": "rgba(171, 99, 250, 0.14)",
    "off_hours": "rgba(128, 128, 128, 0.05)",
}


def _resample_for_display(df: pd.DataFrame, max_candles: int = 1500) -> pd.DataFrame:
    """If there are too many candles to plot smoothly (e.g. 3 months at 5m
    is ~26,000 candles), regroup them into a coarser timeframe *only for
    this visualization*. The backtest always runs on the original data —
    this doesn't touch it, it's purely so the HTML isn't so heavy the
    browser chokes on zoom/pan.
    """
    if len(df) <= max_candles or df.empty:
        return df

    total_minutes = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    target_rule = max(5, total_minutes / max_candles)
    steps = [5, 15, 30, 60, 120, 240, 480, 1440]
    rule_minutes = next((s for s in steps if s >= target_rule), steps[-1])

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for extra in ("hour_utc", "session"):
        if extra in df.columns:
            agg[extra] = "first"

    resampled = (
        df.set_index("timestamp")
        .resample(f"{rule_minutes}min")
        .agg(agg)
        .dropna(subset=["open"])
        .reset_index()
    )
    return resampled


def _resample_generic(df: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Generic downsampling (1 out of every N rows) for DataFrames that
    aren't OHLC — e.g. a channel's bands (Donchian) or other structure
    computed at entry-candle resolution. Doesn't assume which columns `df`
    carries, unlike `_resample_for_display` (which aggregates OHLC/volume)."""
    if len(df) <= max_rows or df.empty:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step].reset_index(drop=True)


def _session_segments(entry_df: pd.DataFrame, max_segments: int = 400) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Collapses the `session` column into contiguous segments (start, end, session)."""
    if entry_df.empty:
        return []
    segments = []
    start_ts = entry_df["timestamp"].iloc[0]
    current_session = entry_df["session"].iloc[0]
    prev_ts = start_ts

    for ts, sess in zip(entry_df["timestamp"].iloc[1:], entry_df["session"].iloc[1:]):
        if sess != current_session:
            segments.append((start_ts, prev_ts, current_session))
            start_ts, current_session = ts, sess
        prev_ts = ts

    segments.append((start_ts, prev_ts, current_session))

    if len(segments) > max_segments:
        # Too many segments to draw comfortably (long dataset): skip the
        # shading instead of overloading the HTML with hundreds of shapes.
        return []
    return segments


def build_scenario_chart(
    entry_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    scenario_name: str,
    pair: str,
    output_path: Path,
) -> Path:
    entry_df = _resample_for_display(entry_df)
    # Generic, in case the "structure" being plotted was computed at entry-
    # candle resolution (e.g. Donchian bands) instead of a lighter higher
    # timeframe (e.g. 1h structure) — without this, a strategy with many
    # entry candles bloats the HTML just like candles did before being
    # resampled.
    structure_df = _resample_generic(structure_df)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{pair} — {scenario_name}", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=entry_df["timestamp"],
            open=entry_df["open"],
            high=entry_df["high"],
            low=entry_df["low"],
            close=entry_df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # Structure's Fibonacci zone (1h), as step lines (changes every time a
    # new swing is confirmed).
    if not structure_df.empty and "zone_lo" in structure_df:
        fig.add_trace(
            go.Scatter(
                x=structure_df["timestamp"], y=structure_df["zone_lo"],
                mode="lines", line=dict(color="rgba(255,167,38,0.6)", width=1, shape="hv"),
                name="Fib zone (0.5)", legendgroup="fib",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=structure_df["timestamp"], y=structure_df["zone_hi"],
                mode="lines", line=dict(color="rgba(255,167,38,0.6)", width=1, shape="hv"),
                name="Fib zone (0.618)", legendgroup="fib",
                fill="tonexty", fillcolor="rgba(255,167,38,0.08)",
            ),
            row=1, col=1,
        )

    # Structure swing highs/lows.
    if not structure_df.empty and "swing_high" in structure_df:
        sh = structure_df[structure_df["swing_high"] == True]  # noqa: E712
        sl = structure_df[structure_df["swing_low"] == True]  # noqa: E712
        fig.add_trace(
            go.Scatter(x=sh["timestamp"], y=sh["high"], mode="markers", name="Swing high (1h)",
                       marker=dict(symbol="triangle-down", color="#ef5350", size=7)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sl["timestamp"], y=sl["low"], mode="markers", name="Swing low (1h)",
                       marker=dict(symbol="triangle-up", color="#26a69a", size=7)),
            row=1, col=1,
        )

    # Trade entries / exits.
    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]

        def _hover(df: pd.DataFrame) -> list[str]:
            return [
                f"{'LONG' if r.direction=='long' else 'SHORT'} · score={r.confidence_score}<br>"
                f"entry={r.entry_price:.2f} exit={r.exit_price:.2f}<br>"
                f"reason={r.exit_reason} · pnl={r.pnl:.2f} ({r.pnl_pct:.2f}%)<br>"
                f"session={r.session} hour_utc={r.hour_utc}"
                for r in df.itertuples()
            ]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(x=longs["entry_time"], y=longs["entry_price"], mode="markers", name="Long entry",
                           marker=dict(symbol="triangle-up", color="#00e676", size=11, line=dict(width=1, color="black")),
                           text=_hover(longs), hoverinfo="text"),
                row=1, col=1,
            )
        if not shorts.empty:
            fig.add_trace(
                go.Scatter(x=shorts["entry_time"], y=shorts["entry_price"], mode="markers", name="Short entry",
                           marker=dict(symbol="triangle-down", color="#ff1744", size=11, line=dict(width=1, color="black")),
                           text=_hover(shorts), hoverinfo="text"),
                row=1, col=1,
            )

        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        if not wins.empty:
            fig.add_trace(
                go.Scatter(x=wins["exit_time"], y=wins["exit_price"], mode="markers", name="Exit (win)",
                           marker=dict(symbol="circle", color="#00e676", size=8, line=dict(width=1, color="black")),
                           text=_hover(wins), hoverinfo="text"),
                row=1, col=1,
            )
        if not losses.empty:
            fig.add_trace(
                go.Scatter(x=losses["exit_time"], y=losses["exit_price"], mode="markers", name="Exit (loss)",
                           marker=dict(symbol="x", color="#ff1744", size=8, line=dict(width=1, color="black")),
                           text=_hover(losses), hoverinfo="text"),
                row=1, col=1,
            )

    # Volume.
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(entry_df["open"], entry_df["close"])]
    fig.add_trace(
        go.Bar(x=entry_df["timestamp"], y=entry_df["volume"], name="Volume", marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    # Session shading.
    for start, end, sess in _session_segments(entry_df):
        fig.add_vrect(
            x0=start, x1=end, fillcolor=SESSION_COLORS.get(sess, "rgba(0,0,0,0.03)"),
            opacity=1, line_width=0, row=1, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def build_equity_curve_chart(equity_curve: pd.DataFrame, scenario_name: str, output_path: Path) -> Path:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve["timestamp"], y=equity_curve["equity"], mode="lines",
            line=dict(color="#26a69a", width=2), fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
            name="Equity",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Equity curve — {scenario_name}",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis_title="Capital",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path
