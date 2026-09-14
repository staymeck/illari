"""Tests for src/backtest/intrabar_entry.py — synthetic 1h/5m data with a
data-dependent fake setup (fires when close >= 105), same convention as
test_evaluate_signal.py's cross-check against real engine internals."""
import pandas as pd

from src.backtest.intrabar_entry import analyze_intrabar_entries
from src.strategies.types import EvalContext, SetupResult
from tests.test_engine_integration import _bar, _df, _strategy


def _fires_above_105(ctx: EvalContext, params: dict) -> SetupResult | None:
    if ctx.price_window["close"].iloc[-1] >= 105:
        return SetupResult(reference_level=100.0)
    return None


def _children(parent_timestamp, rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (open, high, low, close), 20 minutes apart."""
    out = []
    for k, (o, h, l, c) in enumerate(rows):
        out.append({"timestamp": parent_timestamp + pd.Timedelta(minutes=20 * k), "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(out)


def test_detects_a_signal_invisible_to_the_full_close():
    parent = _df(
        [
            _bar(100, 100, 100, 100),  # idx0 padding
            _bar(100, 100, 100, 100),  # idx1
            _bar(100, 100, 100, 100),  # idx2
            _bar(100, 106, 99, 104),  # idx3: closes at 104 -> standard signal never fires (104 < 105)
            _bar(104, 104, 104, 104),  # idx4 trailing
        ]
    )
    children = _children(
        parent["timestamp"].iloc[3],
        [
            (100, 101, 99.5, 100.5),  # cumulative close 100.5 -> no fire
            (100.5, 106, 100, 106),  # cumulative close 106 -> fires here (invisible to the close)
            (106, 106, 99, 104),  # reverses back down by the real close
        ],
    )
    strategy = _strategy(_fires_above_105)

    result = analyze_intrabar_entries(parent, children, strategy)
    row = result[result["bar_index"] == 3].iloc[0]

    assert bool(row["standard_signal"]) is False
    assert bool(row["intrabar_signal"]) is True
    assert row["intrabar_fire_price"] == 106


def test_detects_earlier_entry_when_the_standard_signal_also_fires():
    parent = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 106, 99.5, 106),  # closes at 106 -> standard signal fires too
            _bar(106, 106, 106, 106),
        ]
    )
    children = _children(
        parent["timestamp"].iloc[3],
        [
            (100, 101, 99.5, 100.5),  # no fire yet
            (100.5, 106, 100, 105.5),  # fires here, earlier and at a better (lower) price
            (105.5, 106, 105, 106),  # matches the eventual close
        ],
    )
    strategy = _strategy(_fires_above_105)

    result = analyze_intrabar_entries(parent, children, strategy)
    row = result[result["bar_index"] == 3].iloc[0]

    assert bool(row["standard_signal"]) is True
    assert bool(row["intrabar_signal"]) is True
    assert row["intrabar_fire_price"] == 105.5
    assert row["intrabar_fire_price"] < row["close_price"]


def test_no_signal_anywhere_reports_both_false():
    parent = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 101, 99, 100),  # never gets close to 105
            _bar(100, 100, 100, 100),
        ]
    )
    children = _children(
        parent["timestamp"].iloc[3],
        [
            (100, 100.5, 99.5, 100),
            (100, 101, 99, 100.2),
        ],
    )
    strategy = _strategy(_fires_above_105)

    result = analyze_intrabar_entries(parent, children, strategy)
    row = result[result["bar_index"] == 3].iloc[0]

    assert bool(row["standard_signal"]) is False
    assert bool(row["intrabar_signal"]) is False
    assert row["intrabar_fire_time"] is None


def test_skips_hours_with_no_child_data():
    parent = _df(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 100, 100, 100),
            _bar(100, 106, 99, 106),
            _bar(106, 106, 106, 106),
        ]
    )
    empty_children = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    strategy = _strategy(_fires_above_105)

    result = analyze_intrabar_entries(parent, empty_children, strategy)

    assert result.empty
