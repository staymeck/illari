"""Downloads and caches the crypto Fear & Greed Index from alternative.me — a
free daily proxy for aggregate market sentiment.

Reference: docs/PLAN.md, "Additional data sources".

Usage:
    from src.data.fear_greed import fetch_fear_greed_index

    df = fetch_fear_greed_index(since="2024-06-01", until="2024-12-01")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CACHE_FILE = DATA_DIR / "fear_greed" / "fear_greed_index.parquet"
_API_URL = "https://api.alternative.me/fng/"


def _download_all() -> pd.DataFrame:
    """Downloads the full available history in one request (`limit=0` means
    "all") — the index is small (one value per day since 2018), so there's no
    need to paginate."""
    response = requests.get(_API_URL, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    payload = response.json()["data"]

    df = pd.DataFrame(payload)[["timestamp", "value", "value_classification"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "classification"})
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_fear_greed_index(
    since: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Returns the Fear & Greed Index between `since` and `until`, with an
    on-disk cache (parquet) refreshed whenever it doesn't cover the requested
    range yet (e.g. the first call each day, to pick up new values)."""
    since_ts = pd.Timestamp(since, tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC") if until is not None else pd.Timestamp.now(tz="UTC")

    cached = pd.DataFrame()
    if use_cache and _CACHE_FILE.exists():
        cached = pd.read_parquet(_CACHE_FILE)
        covered = not cached.empty and cached["timestamp"].min() <= since_ts and cached["timestamp"].max() >= until_ts
        if covered:
            mask = (cached["timestamp"] >= since_ts) & (cached["timestamp"] <= until_ts)
            return cached.loc[mask].reset_index(drop=True)

    df = _download_all()
    if use_cache and not df.empty:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE_FILE, index=False)

    mask = (df["timestamp"] >= since_ts) & (df["timestamp"] <= until_ts)
    return df.loc[mask].reset_index(drop=True)
