"""Intra-candle microstructure mining: looks inside each traded (1h) candle
at its finer-grained (5-minute) sub-candles, purely as a statistical
diagnostic - NOT a new entry signal. The lab already decided against
trading at 5-minute resolution (too much noise relative to fixed costs,
see config/markets.py's TIMEFRAMES comment) - this reuses that same data
for a different job: quantifying how much of a traded candle's movement is
genuine net progress versus back-and-forth noise, and whether the candles
our strategy actually enters on look structurally different from the rest
of the market.

Two metrics per parent candle, both deterministic and grounded in Kaufman
(Trading Systems and Methods, ch. 17 - the Efficiency Ratio behind his
Kaufman Adaptive Moving Average), applied here at finer resolution than he
does:

- efficiency_ratio: net change across the candle (|close - open|) divided
  by the sum of absolute step-changes between consecutive child closes
  (starting from the parent's own open). 1.0 = every step moved the same
  direction with no backtracking ("clean"); near 0 = price wandered back
  and forth and barely progressed net ("noisy").
- direction_agreement_pct: % of child candles whose own direction (close
  vs open) agrees with the parent candle's overall direction.

See docs/PLAN.md / Bitácora Illari - motivated directly by the observation
that fixed-timeframe analysis might be missing or misjudging moves that
happen inside a candle ("el mercado no se mueve en pasos discretos de 30
minutos").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def compute_intracandle_metrics(
    parent_df: pd.DataFrame, child_df: pd.DataFrame
) -> pd.DataFrame:
    """Returns parent_df with three columns added: efficiency_ratio,
    direction_agreement_pct, n_children (how many child bars were actually
    found within that parent candle's span - a data-completeness check, not
    a fixed expected count, since gaps happen at either resolution).

    Child bars are assigned to the parent candle whose span contains their
    timestamp: [parent_open_time, next_parent_open_time). Parents with zero
    matching child bars get NaN metrics rather than a fabricated value.
    """
    parent_df = parent_df.reset_index(drop=True)
    child_df = child_df.sort_values("timestamp").reset_index(drop=True)

    parent_times = parent_df["timestamp"].to_numpy()
    if len(parent_times) < 2:
        raise ValueError("need at least 2 parent candles to infer candle span")
    parent_interval = parent_df["timestamp"].iloc[1] - parent_df["timestamp"].iloc[0]

    bucket = np.searchsorted(parent_times, child_df["timestamp"].to_numpy(), side="right") - 1
    child_df = child_df.assign(_parent_idx=bucket)
    child_df = child_df[child_df["_parent_idx"] >= 0]
    # Drop overflow: a child bar after the LAST parent's own span (searchsorted
    # has no upper bound to compare against for the final bucket).
    span_end = pd.Series(parent_times[child_df["_parent_idx"]]) + parent_interval
    child_df = child_df[child_df["timestamp"].to_numpy() < span_end.to_numpy()]

    n = len(parent_df)
    efficiency_ratio = np.full(n, np.nan)
    direction_agreement = np.full(n, np.nan)
    n_children = np.zeros(n, dtype=int)

    for parent_idx, group in child_df.groupby("_parent_idx"):
        parent_open = parent_df["open"].iloc[parent_idx]
        closes = group["close"].to_numpy()
        opens = group["open"].to_numpy()

        prices = np.concatenate([[parent_open], closes])
        steps = np.abs(np.diff(prices))
        path_length = steps.sum()
        net_change = abs(prices[-1] - prices[0])

        if path_length > 0:
            efficiency_ratio[parent_idx] = net_change / path_length

        parent_direction = np.sign(prices[-1] - prices[0])
        if parent_direction != 0:
            child_directions = np.sign(closes - opens)
            direction_agreement[parent_idx] = 100 * (child_directions == parent_direction).mean()

        n_children[parent_idx] = len(group)

    out = parent_df.copy()
    out["efficiency_ratio"] = efficiency_ratio
    out["direction_agreement_pct"] = direction_agreement
    out["n_children"] = n_children
    return out


def compare_groups(metrics_df: pd.DataFrame, is_member: pd.Series, column: str) -> dict:
    """Compares `column` (e.g. "efficiency_ratio") between bars where
    `is_member` is True (e.g. candles a strategy actually entered on) and
    the rest, via a two-sided Mann-Whitney U test - nonparametric, since
    efficiency_ratio is bounded [0, 1] and not assumed normal. Rows with
    NaN in `column` are excluded from both groups first."""
    valid = metrics_df[column].notna()
    is_member = is_member.reindex(metrics_df.index, fill_value=False)
    member = metrics_df.loc[valid & is_member, column]
    rest = metrics_df.loc[valid & ~is_member, column]

    result = {
        "n_member": int(len(member)),
        "n_rest": int(len(rest)),
        "mean_member": round(float(member.mean()), 4) if len(member) else None,
        "median_member": round(float(member.median()), 4) if len(member) else None,
        "mean_rest": round(float(rest.mean()), 4) if len(rest) else None,
        "median_rest": round(float(rest.median()), 4) if len(rest) else None,
        "p_value": None,
    }
    if len(member) >= 2 and len(rest) >= 2:
        test = mannwhitneyu(member, rest, alternative="two-sided")
        result["p_value"] = round(float(test.pvalue), 4)
    return result


def lag1_autocorrelation(returns: pd.Series) -> float | None:
    """Lag-1 autocorrelation of a return series: positive means a move in
    one 5-minute step tends to be followed by another move in the SAME
    direction (locally trending / momentum); negative means it tends to
    reverse (locally choppy / mean-reverting). None if too few points."""
    returns = returns.dropna()
    if len(returns) < 3:
        return None
    value = returns.autocorr(lag=1)
    return round(float(value), 4) if pd.notna(value) else None
