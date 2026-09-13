"""Market-regime measures — objective, independent of the strategy's own
results (see docs/PLAN.md's "Version A vs Version B" distinction: adapting
to a *measured market condition* like volatility or trend strength is
well-grounded — Kaufman, Trading Systems and Methods, ch. 17 "Adaptive
Techniques", most famously his own Adaptive Moving Average (KAMA), which
speeds up in efficient trends and slows down in noise, measured
objectively. Adapting to the strategy's *own* recent win/loss record is a
fundamentally different, much riskier idea — not what this module does.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.adx import adx
from src.analysis.volatility import atr


def volatility_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100) -> float:
    """Where the *current* ATR sits (0-100) within its own trailing
    distribution — a market-relative measure of "is volatility unusually
    high or low right now for this specific market," rather than an
    absolute ATR value that means completely different things on different
    markets/timeframes (the documented weakness of a single fixed ATR
    multiple — see risk/atr_stop.py and docs/PLAN.md's cross-market ATR
    results). Returns NaN if there isn't enough history yet."""
    atr_series = atr(df, period=atr_period)
    recent = atr_series.tail(lookback).dropna()
    if recent.empty:
        return float("nan")
    current = atr_series.iloc[-1]
    if pd.isna(current):
        return float("nan")
    return float((recent <= current).mean() * 100)


def trend_strength_regime(df: pd.DataFrame, adx_period: int = 14, trending_threshold: float = 25.0) -> str:
    """"trending" if the current ADX clears `trending_threshold`, else
    "choppy" — the same underlying measure as confirmations/adx_strength.py,
    exposed as a label for pieces that want to branch on it (e.g. pick a
    different risk multiple) rather than just gate on it."""
    last_adx = adx(df, period=adx_period).iloc[-1]
    if pd.isna(last_adx):
        return "unknown"
    return "trending" if last_adx >= trending_threshold else "choppy"
