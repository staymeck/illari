"""Downloads and caches US equity OHLCV candles via Alpaca's Market Data
API (alpaca-py) — the stock-market equivalent of src/data/fetcher.py's
Binance/ccxt fetcher, same interface (a DataFrame with timestamp/open/
high/low/close/volume) so the rest of the engine (backtest, strategy
pieces, live paper trading) doesn't need to know or care which market a
DataFrame came from.

Needs Alpaca PAPER TRADING API keys (free, no funded account required —
see https://app.alpaca.markets/paper/dashboard/overview) as the
ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY environment variables (or a
local, gitignored .env file — see .env.example). Historical market data
access is the same on paper and live keys; this project only ever reads
history and (eventually) paper-trades, never places live orders.

A real, structural difference from crypto that the rest of the pipeline
does NOT currently account for (flagged here, not hidden): equities only
trade during exchange sessions (weekdays, ~9:30-16:00 ET plus pre/after
market), so the returned series has real gaps (nights, weekends,
holidays) — unlike Binance's 24/7 candles. Swing/structure detection,
which only counts bars (not wall-clock time), still works correctly
across these gaps; anything that assumes a fixed wall-clock cadence
between bars (e.g. the live Docker container's "one candle per hour"
scheduling) would need adjusting before this is used for a live check,
not just a backtest.

Usage:
    from src.data.stock_fetcher import fetch_stock_ohlcv

    df = fetch_stock_ohlcv("AAPL", "1h", since="2023-01-01", until="2023-06-01")
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_TIMEFRAME_RE = re.compile(r"^(\d+)(m|h|d|w)$")
_UNIT_NAMES = {"m": "Minute", "h": "Hour", "d": "Day", "w": "Week"}


def _load_dotenv_once() -> None:
    """Loads a local .env (gitignored) into os.environ if present — cheap
    to call repeatedly, python-dotenv no-ops if the file doesn't exist or
    a var is already set."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _parse_timeframe(timeframe: str):
    """"1h" -> TimeFrame(1, TimeFrameUnit.Hour), "1d" -> TimeFrame(1, Day),
    "5m" -> TimeFrame(5, Minute), etc. — same compact strings the rest of
    this project already uses (config/markets.py's TIMEFRAMES)."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    match = _TIMEFRAME_RE.match(timeframe)
    if not match:
        raise ValueError(f"Unrecognized timeframe {timeframe!r} (expected e.g. '1h', '5m', '1d')")
    amount, unit_letter = match.groups()
    unit = getattr(TimeFrameUnit, _UNIT_NAMES[unit_letter])
    return TimeFrame(int(amount), unit)


def _client():
    _load_dotenv_once()
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = os.environ.get("ALPACA_API_KEY_ID")
    secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set. Copy .env.example to .env "
            "and fill in your Alpaca paper-trading API keys (see that file for where to get them)."
        )
    return StockHistoricalDataClient(api_key, secret_key)


def _cache_path(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / "ohlcv_stocks" / f"{symbol}_{timeframe}.parquet"


def fetch_stock_ohlcv(
    symbol: str,
    timeframe: str,
    since: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Downloads equity OHLCV bars from Alpaca, with an on-disk cache
    (parquet) to avoid repeating downloads across runs — mirrors
    src.data.fetcher.fetch_ohlcv's exact signature and return shape.
    Prices are split-adjusted (not dividend-adjusted) so a stock split
    doesn't show up as a fake, huge single-bar move in the swing/
    structure detection this project's pieces rely on."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest

    since_ts = pd.Timestamp(since, tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC") if until is not None else pd.Timestamp.now(tz="UTC")

    cache_file = _cache_path(symbol, timeframe)
    cached = pd.DataFrame()
    if use_cache and cache_file.exists():
        cached = pd.read_parquet(cache_file)
        covered = not cached.empty and cached["timestamp"].min() <= since_ts and cached["timestamp"].max() >= until_ts
        if covered:
            mask = (cached["timestamp"] >= since_ts) & (cached["timestamp"] <= until_ts)
            return cached.loc[mask].reset_index(drop=True)

    client = _client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_parse_timeframe(timeframe),
        start=since_ts.to_pydatetime(),
        end=until_ts.to_pydatetime(),
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX,  # free tier; SIP (full consolidated tape) needs a paid subscription
    )
    bars = client.get_stock_bars(request)
    rows = bars.data.get(symbol, [])

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(bar.timestamp, tz="UTC"),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in rows
        ]
    )
    if not df.empty:
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df[(df["timestamp"] >= since_ts) & (df["timestamp"] <= until_ts)].reset_index(drop=True)

    if use_cache and not df.empty:
        merged = pd.concat([cached, df]).drop_duplicates(subset="timestamp").sort_values("timestamp")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_file, index=False)

    return df
