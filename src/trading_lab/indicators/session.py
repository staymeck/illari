"""Time-of-day and market session classification (UTC).

Reference sessions (standard, approximate hours widely used in forex/crypto
trading):

  Asia:                00:00 - 08:00 UTC
  London:              08:00 - 16:00 UTC
  New York:            13:00 - 21:00 UTC
  London-NY overlap:   13:00 - 16:00 UTC (highest liquidity of the day)

This classification is purely analytical at this stage: it's used to break
down strategy performance by hour/session in the report, not to filter
which signals are taken.
"""

from __future__ import annotations

import pandas as pd

ASIA_START, ASIA_END = 0, 8
LONDON_START, LONDON_END = 8, 16
NY_START, NY_END = 13, 21
OVERLAP_START, OVERLAP_END = 13, 16


def classify_session(hour_utc: int) -> str:
    if not 0 <= hour_utc <= 23:
        raise ValueError("hour_utc must be between 0 and 23")

    in_overlap = OVERLAP_START <= hour_utc < OVERLAP_END
    in_london = LONDON_START <= hour_utc < LONDON_END
    in_ny = NY_START <= hour_utc < NY_END
    in_asia = ASIA_START <= hour_utc < ASIA_END

    if in_overlap:
        return "overlap_london_ny"
    if in_london and in_ny:
        return "overlap_london_ny"
    if in_london:
        return "london"
    if in_ny:
        return "new_york"
    if in_asia:
        return "asia"
    return "off_hours"


def add_session_columns(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Adds `hour_utc` (0-23) and `session` columns derived from `timestamp_col` (UTC)."""
    out = df.copy()
    ts = out[timestamp_col]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    out["hour_utc"] = ts.dt.hour
    out["session"] = out["hour_utc"].map(classify_session)
    return out
