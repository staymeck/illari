"""Formal statistical read on a strategy's win rate — flagged repeatedly
throughout this session as missing (we kept eyeballing win rates and trade
counts without ever putting a number on how much to trust them). See
docs/PLAN.md / Bitácora Illari.

The question this answers: is the observed win rate plausibly just noise
around this strategy's OWN breakeven rate (given its actual average win/loss
sizes — not a generic R-multiple assumption), or is there a statistically
real edge?

Caveat, stated plainly rather than glossed over: the binomial test below
assumes independent trials. Trades from the same market aren't fully
independent (they share the same underlying trend/regime), so a p-value
here is optimistic, not a rigorous guarantee — it's still far more
informative than eyeballing a win rate with no reference point at all.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


def breakeven_win_rate(trades: pd.DataFrame) -> float:
    """The win rate at which this strategy's own actual average win/loss
    sizes would exactly break even (profit factor == 1) — a strategy-
    specific baseline instead of a generic R-multiple assumption."""
    if trades.empty:
        return 0.5
    wins = trades[trades["pnl_abs"] > 0]
    losses = trades[trades["pnl_abs"] <= 0]
    avg_win = wins["pnl_abs"].mean() if not wins.empty else 0.0
    avg_loss = -losses["pnl_abs"].mean() if not losses.empty else 0.0
    if avg_win + avg_loss == 0:
        return 0.5
    return float(avg_loss / (avg_win + avg_loss))


def significance_report(trades: pd.DataFrame) -> dict:
    """n_trades, observed vs. breakeven win rate, a 95% Wilson confidence
    interval on the true win rate, whether breakeven falls inside it (if
    so, the data can't rule out "no edge"), and a one-sided binomial
    p-value against the breakeven rate."""
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "wins": 0, "observed_win_rate": 0.0, "breakeven_win_rate": 0.0,
            "ci_95_low": 0.0, "ci_95_high": 0.0, "breakeven_inside_ci": True, "p_value_vs_breakeven": None,
        }

    wins = int((trades["pnl_abs"] > 0).sum())
    observed_rate = wins / n
    p0 = breakeven_win_rate(trades)

    ci_low, ci_high = proportion_confint(wins, n, alpha=0.05, method="wilson")
    test = binomtest(wins, n, p0, alternative="greater")

    return {
        "n_trades": n,
        "wins": wins,
        "observed_win_rate": round(observed_rate * 100, 2),
        "breakeven_win_rate": round(p0 * 100, 2),
        "ci_95_low": round(ci_low * 100, 2),
        "ci_95_high": round(ci_high * 100, 2),
        "breakeven_inside_ci": bool(ci_low <= p0 <= ci_high),
        "p_value_vs_breakeven": round(test.pvalue, 4),
    }
