"""Downloads and caches Binance USD-M perpetual futures funding rate history
via ccxt. Funding rate is a free proxy for market-wide long/short bias: a
persistently positive rate means longs are paying shorts (crowd is net long),
and vice versa.

Reference: docs/PLAN.md, "Additional data sources".

Usage:
    from src.data.funding import fetch_funding_rate_history

    df = fetch_funding_rate_history("BTC/USDT", since="2024-06-01", until="2024-12-01")
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Binance caps requests at 1000 entries; funding settles every 8h so this
# covers a wide range per page.
_PAGE_LIMIT = 1000


def _exchange() -> ccxt.binanceusdm:
    return ccxt.binanceusdm({"enableRateLimit": True})


def _cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return DATA_DIR / "funding" / f"{safe_symbol}.parquet"


def fetch_funding_rate_history(
    symbol: str,
    since: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Downloads the funding rate history for a perpetual `symbol` (spot
    notation, e.g. "BTC/USDT" — ccxt resolves it to the linear perpetual on
    binanceusdm), paginating and caching on disk (parquet) like fetch_ohlcv.

    Returns a DataFrame with columns: timestamp, funding_rate.
    """
    since_ts = pd.Timestamp(since, tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC") if until is not None else pd.Timestamp.now(tz="UTC")

    cache_file = _cache_path(symbol)
    cached = pd.DataFrame()
    if use_cache and cache_file.exists():
        cached = pd.read_parquet(cache_file)
        covered = not cached.empty and cached["timestamp"].min() <= since_ts and cached["timestamp"].max() >= until_ts
        if covered:
            mask = (cached["timestamp"] >= since_ts) & (cached["timestamp"] <= until_ts)
            return cached.loc[mask].reset_index(drop=True)

    exchange = _exchange()
    all_rows: list[dict] = []
    cursor_ms = int(since_ts.timestamp() * 1000)
    until_ms = int(until_ts.timestamp() * 1000)

    while cursor_ms < until_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor_ms, limit=_PAGE_LIMIT)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor_ms:
            break  # guard against an infinite loop if the API doesn't advance
        cursor_ms = last_ts + 1
        if len(batch) < _PAGE_LIMIT:
            break  # reached the end of what's available

    df = pd.DataFrame(
        [{"timestamp": row["timestamp"], "funding_rate": row["fundingRate"]} for row in all_rows]
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df[(df["timestamp"] >= since_ts) & (df["timestamp"] <= until_ts)].reset_index(drop=True)

    if use_cache and not df.empty:
        merged = pd.concat([cached, df]).drop_duplicates(subset="timestamp").sort_values("timestamp")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_file, index=False)

    return df
