"""Tests for src/analysis/probability_table.py."""
import numpy as np
import pandas as pd

from src.analysis.probability_table import build_frequency_table, lookup_probability


def _synthetic_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A wandering-but-varied price series (trend + cycle + noise) — enough
    spread in ADX/RSI readings to populate multiple buckets, without
    needing to hand-engineer exact indicator values."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 20, n)
    cycle = 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0, 0.5, n).cumsum() * 0.1
    close = 100 + trend + cycle + noise
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_build_frequency_table_has_expected_shape_and_valid_probabilities():
    df = _synthetic_df()

    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)

    assert {"adx_bucket", "rsi_bucket", "n_samples", "p_up"}.issubset(table.columns)
    assert (table["p_up"] >= 0).all() and (table["p_up"] <= 1).all()
    assert (table["n_samples"] > 0).all()
    assert "adx_edges" in table.attrs and "rsi_edges" in table.attrs


def test_lookup_probability_matches_the_table_for_a_populated_bucket():
    df = _synthetic_df()
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)

    # Pick the most-populated bucket and confirm a lookup for a value inside
    # its range returns exactly that bucket's stored probability.
    biggest = table.sort_values("n_samples", ascending=False).iloc[0]
    adx_interval = table.attrs["adx_edges"][int(biggest["adx_bucket"])]
    rsi_interval = table.attrs["rsi_edges"][int(biggest["rsi_bucket"])]
    adx_value = adx_interval.mid
    rsi_value = rsi_interval.mid

    result = lookup_probability(table, adx_value, rsi_value)

    assert result == biggest["p_up"]


def test_lookup_probability_none_for_missing_values():
    df = _synthetic_df()
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)

    assert lookup_probability(table, float("nan"), 50.0) is None
    assert lookup_probability(table, 20.0, float("nan")) is None


def test_lookup_probability_none_when_bucket_too_thin():
    # A tiny dataset: with 5x5=25 possible buckets and ~10 usable rows,
    # every populated bucket falls well below MIN_SAMPLES (20).
    df = _synthetic_df(n=30)
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)

    adx_interval = table.attrs["adx_edges"][0]
    rsi_interval = table.attrs["rsi_edges"][0]

    assert lookup_probability(table, adx_interval.mid, rsi_interval.mid) is None
