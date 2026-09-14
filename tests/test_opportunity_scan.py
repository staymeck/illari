"""Tests for src/analysis/opportunity_scan.py."""
import pandas as pd

from src.analysis.opportunity_scan import compute_mfe, find_ideal_trades, opportunity_rate


def _df(closes):
    # high/low equal to close keeps the arithmetic easy to hand-check.
    return pd.DataFrame({"high": closes, "low": closes, "close": closes})


def _df_with_ts(closes):
    df = _df(closes)
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(closes), freq="h")
    return df


def test_compute_mfe_known_forward_extremes():
    # close: 10, 20, 30, 40, 50 - horizon 3.
    df = _df([10, 20, 30, 40, 50])

    result = compute_mfe(df, horizon_bars=3)

    # Only rows with a full 3-bar forward window survive: index 0 and 1.
    assert list(result.index) == [0, 1]
    # idx0: forward window is closes[1:4] = 20,30,40 -> best long = 40 vs entry 10 -> 300%.
    assert result.loc[0, "mfe_long_pct"] == 300.0
    # idx0: best short = entry 10 vs min(20,30,40)=20 -> negative (price only rose).
    assert result.loc[0, "mfe_short_pct"] < 0
    # idx1: forward window is closes[2:5] = 30,40,50 -> best long = 50 vs entry 20 -> 150%.
    assert result.loc[1, "mfe_long_pct"] == 150.0


def test_compute_mfe_drops_incomplete_tail():
    df = _df([10, 20, 30])

    result = compute_mfe(df, horizon_bars=3)

    assert result.empty


def test_opportunity_rate_all_bars_clear_a_low_threshold():
    # A steadily rising series: every bar has a long opportunity well above
    # a tiny threshold, and no short opportunity at all.
    df = _df(list(range(10, 210, 10)))  # 10, 20, ..., 200

    report = opportunity_rate(df, horizon_bars=2, threshold_pct=0.1)

    assert report["n_bars"] > 0
    assert report["long_opportunity_pct"] == 100.0
    assert report["short_opportunity_pct"] == 0.0
    assert report["either_opportunity_pct"] == 100.0


def test_opportunity_rate_empty_when_no_full_window_exists():
    df = _df([10, 20])

    report = opportunity_rate(df, horizon_bars=5, threshold_pct=1.0)

    assert report["n_bars"] == 0
    assert report["long_opportunity_pct"] == 0.0


def test_find_ideal_trades_detects_one_clear_long_swing():
    # Flat, then a clean rise from 100 to 150, then flat again: exactly one
    # long peak, no shorts (nothing ever drops enough to clear the threshold).
    closes = [100] * 5 + [110, 120, 130, 140, 150] + [150] * 10
    df = _df_with_ts(closes)

    ideal = find_ideal_trades(df, horizon_bars=5, threshold_pct=5.0, min_distance_bars=3)

    assert (ideal["direction"] == "short").sum() == 0
    longs = ideal[ideal["direction"] == "long"]
    assert len(longs) >= 1
    # Every detected long trade must be a genuine profitable round trip.
    assert (longs["exit_price"] > longs["entry_price"]).all()


def test_find_ideal_trades_respects_min_distance():
    # A repeating small up-down oscillation produces many closely-spaced
    # local peaks in the MFE series - a large min_distance should collapse
    # that into noticeably fewer of them than a permissive one.
    cycle = [0, 2, 4, 2, 0, -2]
    closes = [100 + v for v in cycle * 15]  # 90 bars, ~15 repetitions
    df = _df_with_ts(closes)

    loose = find_ideal_trades(df, horizon_bars=3, threshold_pct=0.5, min_distance_bars=1)
    tight = find_ideal_trades(df, horizon_bars=3, threshold_pct=0.5, min_distance_bars=20)

    loose_longs = len(loose[loose["direction"] == "long"]) if not loose.empty else 0
    tight_longs = len(tight[tight["direction"] == "long"]) if not tight.empty else 0

    assert loose_longs > tight_longs
