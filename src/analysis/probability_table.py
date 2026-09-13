"""Historical frequency table: an explicitly simple, "just to see results"
alternative to the deterministic context pieces (dow_trend/ma_trend) — see
docs/PLAN.md.

IMPORTANT — this is NOT a validated model. The table is built from the same
historical window it will be evaluated against (in-sample lookup), which is
fine for a first look at whether ADX/RSI conditions carry any information
at all, but it is not walk-forward validated. Before trusting any result
from this, the table must be rebuilt on an earlier fold only and evaluated
on a later, untouched fold — exactly the same discipline already applied to
every other strategy in this lab (see docs/PLAN.md's walk-forward section).
Treat a promising result here as "worth investigating further," not as
evidence of an edge.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.adx import adx
from src.analysis.momentum import rsi

MIN_SAMPLES = 20  # buckets thinner than this are reported as "unknown", not guessed


def build_frequency_table(
    df: pd.DataFrame, horizon_bars: int = 5, n_buckets: int = 5, adx_period: int = 14, rsi_period: int = 14
) -> pd.DataFrame:
    """For every historical bar, buckets ADX and RSI into `n_buckets` bins
    each (roughly equal-sized, via quantiles) and records whether `close`
    was higher `horizon_bars` later. Returns one row per (adx_bucket,
    rsi_bucket) combination actually observed, with the sample count and
    the empirical P(up)."""
    adx_values = adx(df, period=adx_period)
    rsi_values = rsi(df["close"], period=rsi_period)
    future_up = (df["close"].shift(-horizon_bars) > df["close"]).astype(float)

    working = pd.DataFrame({"adx": adx_values, "rsi": rsi_values, "future_up": future_up}).dropna()

    working["adx_bucket"] = pd.qcut(working["adx"], q=n_buckets, labels=False, duplicates="drop")
    working["rsi_bucket"] = pd.qcut(working["rsi"], q=n_buckets, labels=False, duplicates="drop")

    grouped = working.groupby(["adx_bucket", "rsi_bucket"])["future_up"]
    table = grouped.agg(n_samples="count", p_up="mean").reset_index()

    # Bucket edges are needed at lookup time to classify a *new* value into
    # the same bins the table was built with.
    adx_edges = pd.qcut(working["adx"], q=n_buckets, duplicates="drop").cat.categories
    rsi_edges = pd.qcut(working["rsi"], q=n_buckets, duplicates="drop").cat.categories
    table.attrs["adx_edges"] = adx_edges
    table.attrs["rsi_edges"] = rsi_edges
    return table


def _bucket_of(value: float, edges) -> int | None:
    for i, interval in enumerate(edges):
        if value in interval:
            return i
    if len(edges) and value <= edges[0].left:
        return 0
    if len(edges) and value >= edges[-1].right:
        return len(edges) - 1
    return None


def lookup_probability(table: pd.DataFrame, adx_value: float, rsi_value: float) -> float | None:
    """Returns the empirical P(up) for the bucket `(adx_value, rsi_value)`
    falls into, or None if that bucket doesn't exist in the table or has
    fewer than MIN_SAMPLES observations — an explicit "don't know" rather
    than a number fabricated from too little data."""
    if pd.isna(adx_value) or pd.isna(rsi_value):
        return None

    adx_edges = table.attrs.get("adx_edges")
    rsi_edges = table.attrs.get("rsi_edges")
    if adx_edges is None or rsi_edges is None:
        raise ValueError("table is missing bucket edges — build it with build_frequency_table()")

    adx_bucket = _bucket_of(adx_value, adx_edges)
    rsi_bucket = _bucket_of(rsi_value, rsi_edges)
    if adx_bucket is None or rsi_bucket is None:
        return None

    row = table[(table["adx_bucket"] == adx_bucket) & (table["rsi_bucket"] == rsi_bucket)]
    if row.empty or row["n_samples"].iloc[0] < MIN_SAMPLES:
        return None
    return float(row["p_up"].iloc[0])
