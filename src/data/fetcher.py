"""Downloads and caches Binance (spot) OHLCV candles via ccxt.

Usage:
    from src.data.fetcher import fetch_ohlcv, earliest_available

    df = fetch_ohlcv("BTC/USDT", "1h", since="2023-01-01", until="2023-06-01")
    first_candle_ts = earliest_available("SOL/USDT", "1d")
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Binance caps requests at 1000 candles; we paginate to cover the full requested range.
_PAGE_LIMIT = 1000


def _exchange() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True})


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return DATA_DIR / "ohlcv" / f"{safe_symbol}_{timeframe}.parquet"


def earliest_available(symbol: str, timeframe: str = "1d") -> pd.Timestamp:
    """Date of the first candle available for `symbol` on Binance — i.e. the
    approximate real listing date, needed before fixing equal-length scenario
    windows across all markets (see docs/PLAN.md, listing-date caveat)."""
    exchange = _exchange()
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=0, limit=1)
    if not candles:
        raise ValueError(f"Binance returned no candles for {symbol} ({timeframe})")
    return pd.to_datetime(candles[0][0], unit="ms", utc=True)


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Downloads OHLCV candles by paginating the Binance API, with an on-disk
    cache (parquet) to avoid repeating downloads across runs."""
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

    exchange = _exchange()
    all_rows: list[list] = []
    cursor_ms = int(since_ts.timestamp() * 1000)
    until_ms = int(until_ts.timestamp() * 1000)

    while cursor_ms < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor_ms, limit=_PAGE_LIMIT)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor_ms:
            break  # guard against an infinite loop if the API doesn't advance
        cursor_ms = last_ts + 1
        if len(batch) < _PAGE_LIMIT:
            break  # reached the end of what's available

    df = pd.DataFrame(all_rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns="ts_ms").drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df[(df["timestamp"] >= since_ts) & (df["timestamp"] <= until_ts)].reset_index(drop=True)

    if use_cache and not df.empty:
        merged = pd.concat([cached, df]).drop_duplicates(subset="timestamp").sort_values("timestamp")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_file, index=False)

    return df
