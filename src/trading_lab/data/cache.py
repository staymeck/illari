"""Local parquet cache of OHLCV candles, so we don't re-download on every run.

One file per combination (exchange, pair, timeframe, scenario, exact date
range). The date range is deliberately part of the key: if a scenario's
dates change while keeping the same name, a stale cache covering a
different period must NOT be silently reused.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_lab.data.fetcher import fetch_ohlcv_range

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)


def _date_tag(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d")


def cache_path(
    exchange_id: str,
    pair: str,
    timeframe: str,
    scenario_name: str,
    start: datetime,
    end: datetime,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    fname = (
        f"{_safe(exchange_id)}_{_safe(pair)}_{_safe(timeframe)}_{_safe(scenario_name)}"
        f"_{_date_tag(start)}_{_date_tag(end)}.parquet"
    )
    return cache_dir / fname


def get_ohlcv(
    exchange_id: str,
    pair: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    scenario_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Returns the candles for the range, using the on-disk cache if it exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(exchange_id, pair, timeframe, scenario_name, start, end, cache_dir)

    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    df = fetch_ohlcv_range(pair, timeframe, start, end, exchange_id=exchange_id)
    df.to_parquet(path, index=False)
    return df
