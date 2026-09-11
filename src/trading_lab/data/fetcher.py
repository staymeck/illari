"""Downloads historical OHLCV candles from an exchange (via ccxt).

All time handling is in UTC / epoch milliseconds, which is what ccxt uses
internally. Pagination is necessary because exchanges limit how many
candles are returned per request (Binance: 1000 per call).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _to_utc_timestamp(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fetch_ohlcv_range(
    pair: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    exchange_id: str = "binance",
    limit: int = 1000,
    max_retries: int = 3,
    retry_sleep_s: float = 1.5,
) -> pd.DataFrame:
    """Downloads every candle for `pair`/`timeframe` between `start` and `end` (UTC).

    Paginates automatically until the whole range is covered. Returns a
    DataFrame sorted by ascending timestamp, with `timestamp` as a
    tz-aware UTC datetime, with no duplicates.
    """
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})

    since_ms = _to_ms(start)
    end_ms = _to_ms(end)

    all_rows: list[list] = []
    cursor = since_ms

    while cursor < end_ms:
        batch = None
        for attempt in range(max_retries):
            try:
                batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=cursor, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to download {pair} {timeframe} from {cursor}: {exc}"
                    ) from exc
                time.sleep(retry_sleep_s * (attempt + 1))

        if not batch:
            break

        all_rows.extend(batch)

        last_ts = batch[-1][0]
        if last_ts <= cursor:
            # The exchange didn't move forward (end of available data); avoids an infinite loop.
            break
        cursor = last_ts + 1

        if len(batch) < limit:
            # Last page was partial: no more data in this range.
            break

    if not all_rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS).astype(
            {"timestamp": "datetime64[ns, UTC]"}
        )

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    start_ts, end_ts = _to_utc_timestamp(start), _to_utc_timestamp(end)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
    return df.reset_index(drop=True)
