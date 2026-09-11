"""Probability analysis: "given this condition held, what did the market
do afterward" — evaluated per ingredient on its own and for the strategy's
full signal, against a comparison baseline.

Important — this is NOT lookahead bias: `forward_return`/`forward_direction`
look ahead *on purpose*, to VALIDATE after the fact how good a condition
that already occurred at instant `t` turned out to be. The signal itself
(`confluence_strategy.py`) never uses these values to decide anything in
real time — it's exactly the same separation that already exists between
"opening a trade at `t`" (without looking ahead) and "measuring its
outcome" (which necessarily looks at candles after `t`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading_lab.indicators import candle_geometry, volume as volume_module
from trading_lab.strategy.base import Signal

SIGNIFICANCE_ALPHA = 0.05


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, via `math.erf` — no scipy dependency needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sample proportion test (z-test): is the "success" proportion of
    group 1 (x1 of n1) different from group 2's (x2 of n2)? Returns
    (z, p_value), two-tailed. A small `p_value` means the observed
    difference is unlikely if both groups had the same real proportion
    (i.e. there's evidence the condition does change something, beyond
    sample noise)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    variance = p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / math.sqrt(variance)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return z, p_value


def one_sample_proportion_z_test(x: int, n: int, p0: float) -> tuple[float, float]:
    """One-sample test: is the observed proportion (x of n) different from
    a base proportion `p0` (e.g. the probability that the market rises/
    falls over that horizon, unconditionally)?"""
    if n == 0 or p0 <= 0 or p0 >= 1:
        return float("nan"), float("nan")
    p_hat = x / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan"), float("nan")
    z = (p_hat - p0) / se
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return z, p_value


def forward_return(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """% return from each candle's close to `horizon_bars` candles later.
    The last `horizon_bars` rows stay NaN (there aren't enough future
    candles in the dataset to measure them)."""
    future_close = df["close"].shift(-horizon_bars)
    return (future_close / df["close"] - 1.0) * 100.0


def forward_direction(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """Sign of the forward return: 1 (up), -1 (down), 0 (flat), NaN if
    there isn't enough future data."""
    return np.sign(forward_return(df, horizon_bars))


def condition_probability(
    df: pd.DataFrame,
    condition: pd.Series,
    horizon_bars: int,
    baseline: pd.Series | None = None,
) -> dict:
    """P(price rises within `horizon_bars`) given `condition`, against a
    comparison baseline (`baseline`, or the whole dataset if not passed).
    `condition`/`baseline` are booleans aligned by index to `df`.
    """
    fwd_ret = forward_return(df, horizon_bars)
    has_future = fwd_ret.notna()

    condition = condition.fillna(False).astype(bool)
    cond_mask = condition & has_future
    base_mask = (baseline.fillna(False).astype(bool) if baseline is not None else pd.Series(True, index=df.index)) & has_future
    # Complement (condition false) restricted to rows with future data —
    # this is the statistically correct comparison for the significance
    # test (it doesn't overlap with the condition's own group, unlike
    # "base_mask" which defaults to the whole dataset).
    complement_mask = (~condition) & has_future

    n = int(cond_mask.sum())
    n_base = int(base_mask.sum())
    n_complement = int(complement_mask.sum())

    p_up_condition = float((fwd_ret[cond_mask] > 0).mean() * 100.0) if n > 0 else float("nan")
    p_up_baseline = float((fwd_ret[base_mask] > 0).mean() * 100.0) if n_base > 0 else float("nan")
    avg_return_condition = float(fwd_ret[cond_mask].mean()) if n > 0 else float("nan")
    avg_return_baseline = float(fwd_ret[base_mask].mean()) if n_base > 0 else float("nan")

    edge_pp = p_up_condition - p_up_baseline if n > 0 and n_base > 0 else float("nan")

    x_condition = int((fwd_ret[cond_mask] > 0).sum())
    x_complement = int((fwd_ret[complement_mask] > 0).sum())
    _, p_value = two_proportion_z_test(x_condition, n, x_complement, n_complement)

    return {
        "n": n,
        "p_up_condition_pct": round(p_up_condition, 2) if n > 0 else None,
        "p_up_base_pct": round(p_up_baseline, 2) if n_base > 0 else None,
        "edge_pp": round(edge_pp, 2) if not pd.isna(edge_pp) else None,
        "avg_return_condition_pct": round(avg_return_condition, 3) if n > 0 else None,
        "avg_return_base_pct": round(avg_return_baseline, 3) if n_base > 0 else None,
        "p_value": round(p_value, 4) if not pd.isna(p_value) else None,
        "significant_5pct": bool(not pd.isna(p_value) and p_value < SIGNIFICANCE_ALPHA),
    }


def _session_conditions(merged: pd.DataFrame) -> dict[str, pd.Series]:
    if "session" not in merged.columns:
        return {}
    return {f"session_{s}": (merged["session"] == s) for s in sorted(merged["session"].dropna().unique())}


def build_probability_table(
    merged: pd.DataFrame,
    horizon_bars: int,
    trendline_proximity_atr_mult: float = 1.0,
) -> pd.DataFrame:
    """Builds the per-ingredient probability table over the already-aligned
    multi-timeframe DataFrame (`merged`, output of
    `mtf_align.align_multi_timeframe`/`align_higher_timeframe` with each
    indicator's columns already computed — see `<strategy>.prepare_*`).

    Not every strategy computes the same columns (e.g. Donchian Breakout
    has no 1h structure or candle patterns) — each ingredient is only
    evaluated if its column(s) exist in `merged`, so this function works
    for any registered strategy.
    """
    pressure = volume_module.buy_sell_pressure_proxy(merged)

    conditions: dict[str, pd.Series] = {}
    if "bias_trend" in merged.columns:
        conditions["bullish_bias_1D"] = merged["bias_trend"] == "up"
        conditions["bearish_bias_1D"] = merged["bias_trend"] == "down"
    if "struct_trend" in merged.columns:
        conditions["bullish_structure_1h"] = merged["struct_trend"] == "up"
        conditions["bearish_structure_1h"] = merged["struct_trend"] == "down"
    if "confirm_bullish" in merged.columns:
        conditions["bullish_candle_pattern_5m"] = merged["confirm_bullish"] == True  # noqa: E712
    if "confirm_bearish" in merged.columns:
        conditions["bearish_candle_pattern_5m"] = merged["confirm_bearish"] == True  # noqa: E712
    if "volume_ok" in merged.columns:
        conditions["sufficient_volume"] = merged["volume_ok"] == True  # noqa: E712
    if "breakout_up" in merged.columns:
        conditions["bullish_channel_breakout"] = merged["breakout_up"] == True  # noqa: E712
    if "breakout_down" in merged.columns:
        conditions["bearish_channel_breakout"] = merged["breakout_down"] == True  # noqa: E712

    conditions["buying_pressure"] = pressure > 0
    conditions["selling_pressure"] = pressure < 0
    conditions.update(_session_conditions(merged))

    # Individual candle geometry (always computable from the entry
    # candle's raw OHLC, regardless of strategy): strong conviction
    # (marubozu), range compression/expansion relative to its own recent
    # history (NR7/WR7), and consolidation (inside bar).
    bullish_candle = merged["close"] > merged["open"]
    marubozu = candle_geometry.is_marubozu(merged)
    conditions["bullish_marubozu"] = marubozu & bullish_candle
    conditions["bearish_marubozu"] = marubozu & ~bullish_candle
    conditions["narrow_range_nr7"] = candle_geometry.is_narrow_range(merged)
    conditions["wide_range_wr7"] = candle_geometry.is_wide_range(merged)
    conditions["inside_bar"] = candle_geometry.is_inside_bar(merged)

    if "struct_dist_to_support" in merged.columns and "atr" in merged.columns:
        conditions["near_support_line"] = merged["struct_dist_to_support"].notna() & (
            merged["struct_dist_to_support"].abs() <= merged["atr"] * trendline_proximity_atr_mult
        )
    if "struct_dist_to_resistance" in merged.columns and "atr" in merged.columns:
        conditions["near_resistance_line"] = merged["struct_dist_to_resistance"].notna() & (
            merged["struct_dist_to_resistance"].abs() <= merged["atr"] * trendline_proximity_atr_mult
        )

    rows = []
    for name, cond in conditions.items():
        result = condition_probability(merged, cond.fillna(False), horizon_bars)
        rows.append({"ingredient": name, **result})

    return pd.DataFrame(rows)


def signal_hit_probability(signals: list[Signal], entry_df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """For the strategy's real signals (direction-aware): on a long, did
    price rise by the horizon?; on a short, did it fall? Empirical hit
    probability of the FULL signal — to compare against each isolated
    ingredient from `build_probability_table`.

    The significance test compares the hit rate against the market's BASE
    probability over that horizon (P(up) for long, P(down) for short, over
    the whole `entry_df`) — not against 50%, because the market can have
    its own directional drift (e.g. a sustained uptrend) that has nothing
    to do with whether the strategy itself has an edge.
    """
    df = entry_df.reset_index(drop=True)
    ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}
    fwd_ret = forward_return(df, horizon_bars)
    valid_ret = fwd_ret.dropna()

    baseline_p_up = float((valid_ret > 0).mean()) if len(valid_ret) > 0 else float("nan")
    baseline_by_direction = {"long": baseline_p_up, "short": 1.0 - baseline_p_up if not pd.isna(baseline_p_up) else float("nan")}

    rows_by_direction: dict[str, list[float]] = {"long": [], "short": []}
    for sig in signals:
        idx = ts_to_idx.get(sig.timestamp)
        if idx is None:
            continue
        ret = fwd_ret.iloc[idx]
        if pd.isna(ret):
            continue
        # Return "in favor of" the signal's direction: positive = hit.
        favorable_ret = ret if sig.direction == "long" else -ret
        rows_by_direction[sig.direction].append(favorable_ret)

    rows = []
    for direction, returns in rows_by_direction.items():
        n = len(returns)
        p0 = baseline_by_direction[direction]
        if n == 0:
            rows.append(
                {
                    "direction": direction, "n": 0, "hit_rate_pct": None, "avg_return_pct": None,
                    "p_base_pct": round(p0 * 100.0, 2) if not pd.isna(p0) else None,
                    "p_value": None, "significant_5pct": False,
                }
            )
            continue
        arr = np.array(returns)
        x = int((arr > 0).sum())
        _, p_value = one_sample_proportion_z_test(x, n, p0) if not pd.isna(p0) else (float("nan"), float("nan"))
        rows.append(
            {
                "direction": direction,
                "n": n,
                "hit_rate_pct": round(float((arr > 0).mean() * 100.0), 2),
                "avg_return_pct": round(float(arr.mean()), 3),
                "p_base_pct": round(p0 * 100.0, 2) if not pd.isna(p0) else None,
                "p_value": round(p_value, 4) if not pd.isna(p_value) else None,
                "significant_5pct": bool(not pd.isna(p_value) and p_value < SIGNIFICANCE_ALPHA),
            }
        )
    return pd.DataFrame(rows)
