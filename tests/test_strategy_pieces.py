"""Wiring tests for src/strategies/{context,setups,confirmations,risk}/*.py —
confirm each piece correctly delegates to the already-tested analysis
functions it wraps (see tests/test_structure.py, test_fibonacci.py,
test_candles.py, test_volume.py for the underlying logic), not re-deriving
that logic here."""
import pandas as pd

from src.analysis.structure import find_swing_points
from src.strategies.confirmations.candlestick import candlestick
from src.strategies.confirmations.fibonacci import fibonacci_confluence
from src.strategies.confirmations.volume import volume as volume_confirmation
from src.strategies.context.dow_trend import dow_trend
from src.strategies.risk.fixed_pct import fixed_pct_stop, risk_reward_target
from src.strategies.setups.support_touch import support_touch
from src.strategies.types import EvalContext, SetupResult


def _candles(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    out = []
    for r in rows:
        o, c = r["open"], r["close"]
        out.append(
            {
                "timestamp": None,
                "open": o,
                "close": c,
                "high": r.get("high", max(o, c)),
                "low": r.get("low", min(o, c)),
                "volume": r.get("volume", 1.0),
            }
        )
    df = pd.DataFrame(out)
    df["timestamp"] = idx
    return df


def _ctx(df: pd.DataFrame, order: int = 2) -> EvalContext:
    marked = find_swing_points(df, order=order)
    return EvalContext(price_window=df, marked_window=marked)


def test_dow_trend_piece_delegates_to_classify_trend():
    closes = [1, 2, 1.5, 3, 2.5, 5, 4, 7]
    df = _candles([{"open": c, "close": c} for c in closes])

    trend = dow_trend(_ctx(df, order=1), {"order": 1, "lookback_swings": 2})

    assert trend == "uptrend"


def test_support_touch_piece_returns_matched_level_and_extras():
    # Two lows near 3.0 (cluster -> level ~3.005), price now sitting just
    # above it (support_touch only matches from above, i.e. a pullback down
    # onto support, not a break below it).
    closes = [5, 3, 5, 3.01, 5, 8, 5, 3.02]
    df = _candles([{"open": c, "close": c} for c in closes])

    result = support_touch(_ctx(df, order=1), {"order": 1, "tolerance_pct": 1.0})

    assert isinstance(result, SetupResult)
    assert 2.99 <= result.reference_level <= 3.01
    assert "support_level" in result.extras


def test_support_touch_piece_returns_none_without_a_nearby_level():
    closes = [5, 3, 5, 3.01, 5, 8, 5, 8.0]  # last close far from the support cluster
    df = _candles([{"open": c, "close": c} for c in closes])

    result = support_touch(_ctx(df, order=1), {"order": 1, "tolerance_pct": 1.0})

    assert result is None


def test_fibonacci_confirmation_piece_returns_ratio_extra():
    # low=1 (idx3) then high=10 (idx7), then a monotonic decline (no new
    # confirmed low pivot in between) down to the 61.8% retracement of
    # [1, 10] = 10 - 9*0.618 = 4.438.
    closes = [5, 4, 3, 1, 3, 5, 7, 10, 8, 6, 5, 4.44]
    df = _candles([{"open": c, "close": c} for c in closes])

    result = fibonacci_confluence(_ctx(df, order=2), {"order": 2, "tolerance_pct": 1.0})

    assert result is not None
    assert result.name == "fibonacci"
    assert "fib_ratio" in result.extras


def test_candlestick_confirmation_piece_detects_hammer():
    df = _candles(
        [
            {"open": 12, "close": 11, "high": 12.1, "low": 10},
            {"open": 9, "close": 10, "high": 10.1, "low": 5},  # hammer
        ]
    )

    result = candlestick(_ctx(df), {})

    assert result is not None
    assert result.extras["pattern"] == "hammer"


def test_candlestick_confirmation_piece_none_for_plain_candle():
    df = _candles([{"open": 10, "close": 10.2, "high": 10.5, "low": 9.5}])

    assert candlestick(_ctx(df), {}) is None


def test_volume_confirmation_piece_true_when_spike_and_buyer_dominant():
    rows = [{"open": 1, "close": 2, "volume": 10}] * 3
    rows += [{"open": 2, "close": 1, "volume": 5}]
    rows += [{"open": 1, "close": 2, "volume": 60}]  # spike, still net buyer-dominant
    df = _candles(rows)

    result = volume_confirmation(_ctx(df), {"window": 4, "spike_threshold": 1.2, "min_bias": 0.1})

    assert result is not None
    assert "volume_bias" in result.extras


def test_volume_confirmation_piece_none_without_spike():
    df = _candles([{"open": 1, "close": 2, "volume": 10}] * 5)

    assert volume_confirmation(_ctx(df), {"window": 3}) is None


def test_fixed_pct_stop_matches_expected_formula():
    setup = SetupResult(reference_level=100.0)
    ctx = _ctx(_candles([{"open": 1, "close": 1}]))

    stop = fixed_pct_stop(entry_price=101.0, setup=setup, ctx=ctx, params={"pct_below_reference": 0.5})

    assert stop == 100.0 * (1 - 0.5 / 100)


def test_risk_reward_target_matches_expected_formula():
    ctx = _ctx(_candles([{"open": 1, "close": 1}]))

    target = risk_reward_target(entry_price=100.0, stop_price=98.0, ctx=ctx, params={"ratio": 2.0})

    assert target == 100.0 + 2.0 * (100.0 - 98.0)
