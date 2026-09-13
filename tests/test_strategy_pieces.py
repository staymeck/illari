"""Wiring tests for src/strategies/{context,setups,confirmations,risk}/*.py —
confirm each piece correctly delegates to the already-tested analysis
functions it wraps (see tests/test_structure.py, test_fibonacci.py,
test_candles.py, test_volume.py for the underlying logic), not re-deriving
that logic here."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.structure import find_swing_points
from src.strategies.confirmations.adx_strength import adx_strength
from src.strategies.confirmations.candlestick import candlestick
from src.strategies.confirmations.fibonacci import fibonacci_confluence
from src.strategies.confirmations.macd_momentum import macd_momentum
from src.strategies.confirmations.rsi_momentum import rsi_momentum
from src.strategies.confirmations.session_filter import session_filter
from src.strategies.confirmations.volume import volume as volume_confirmation
from src.strategies.confirmations.vwap_bias import vwap_bias
from src.strategies.context.dow_trend import dow_trend
from src.strategies.context.ma_trend import ma_trend
from src.strategies.context.probability_trend import probability_trend
from src.strategies.risk.atr_stop import atr_stop
from src.strategies.risk.fixed_pct import fixed_pct_stop, risk_reward_target
from src.strategies.setups.breakout import breakout
from src.strategies.setups.mean_reversion import mean_reversion
from src.strategies.setups.scheduled_entry import scheduled_entry
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


def test_ma_trend_piece_delegates_to_ma_cross_trend():
    df = pd.DataFrame({"close": list(range(1, 101))})  # strong, sustained rally
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    trend = ma_trend(ctx, {"fast": 5, "slow": 20, "method": "ema"})

    assert trend == "uptrend"


def test_breakout_piece_fires_above_prior_high():
    highs = [10, 11, 9, 10, 10]
    df = pd.DataFrame(
        {
            "open": highs + [14],
            "close": highs + [15],
            "high": highs + [15],
            "low": highs + [14],
            "volume": [1.0] * 6,
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = breakout(ctx, {"lookback": 5})

    assert isinstance(result, SetupResult)
    assert result.reference_level == 11  # highest prior high
    assert result.extras["breakout_level"] == 11


def test_breakout_piece_none_when_not_breaking_out():
    highs = [10, 11, 9, 10, 10]
    df = pd.DataFrame(
        {
            "open": highs + [10],
            "close": highs + [10.5],  # doesn't clear the prior high of 11
            "high": highs + [10.5],
            "low": highs + [10],
            "volume": [1.0] * 6,
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert breakout(ctx, {"lookback": 5}) is None


def test_mean_reversion_piece_fires_below_lower_band():
    closes = [10.0] * 9 + [1.0]  # sharp drop below the lower Bollinger Band
    df = pd.DataFrame({"open": closes, "close": closes, "high": closes, "low": closes, "volume": [1.0] * 10})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = mean_reversion(ctx, {"window": 10, "num_std": 1.0})

    assert isinstance(result, SetupResult)
    assert "bollinger_lower" in result.extras


def test_mean_reversion_piece_none_within_the_bands():
    # Small, ordinary fluctuation that stays inside the bands — not the
    # degenerate case of a perfectly flat series (std=0 would make the
    # lower band equal the price itself, trivially "touching" it).
    closes = [10.0, 10.2, 9.9, 10.1, 10.0, 9.95, 10.05, 10.0, 10.1, 10.05]
    df = pd.DataFrame({"open": closes, "close": closes, "high": closes, "low": closes, "volume": [1.0] * 10})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert mean_reversion(ctx, {"window": 10, "num_std": 2.0}) is None


def test_rsi_momentum_confirmation_piece_passes_on_strong_uptrend():
    df = pd.DataFrame({"close": list(range(1, 30))})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = rsi_momentum(ctx, {"period": 14, "min_rsi": 50})

    assert result is not None
    assert result.extras["rsi"] == 100.0


def test_rsi_momentum_confirmation_piece_none_on_downtrend():
    df = pd.DataFrame({"close": list(range(30, 1, -1))})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert rsi_momentum(ctx, {"period": 14, "min_rsi": 50}) is None


def test_macd_momentum_confirmation_piece_passes_on_sustained_uptrend():
    df = pd.DataFrame({"close": list(range(1, 80))})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = macd_momentum(ctx, {})

    assert result is not None
    assert result.extras["macd_histogram"] > 0


def test_macd_momentum_confirmation_piece_none_on_flat_series():
    df = pd.DataFrame({"close": [100.0] * 40})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert macd_momentum(ctx, {}) is None


def test_vwap_bias_confirmation_piece_passes_above_vwap():
    df = pd.DataFrame(
        {
            "high": [12.0, 22.0],
            "low": [8.0, 18.0],
            "close": [10.0, 25.0],
            "volume": [1.0, 3.0],
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = vwap_bias(ctx, {"window": 2, "min_bias_pct": 0.0})

    assert result is not None
    assert result.extras["vwap_bias_pct"] > 0


def test_vwap_bias_confirmation_piece_none_below_threshold():
    df = pd.DataFrame({"high": [10.0], "low": [10.0], "close": [10.0], "volume": [1.0]})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert vwap_bias(ctx, {"window": 5, "min_bias_pct": 0.0}) is None


def test_atr_stop_piece_below_entry_by_atr_multiple():
    df = pd.DataFrame(
        {
            "high": [102.0] * 20,
            "low": [100.0] * 20,
            "close": [101.0] * 20,
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())
    setup = SetupResult(reference_level=100.0)  # unused by this risk piece

    stop = atr_stop(entry_price=110.0, setup=setup, ctx=ctx, params={"period": 14, "multiple": 1.5})

    assert stop == 110.0 - 1.5 * 2.0  # true range settles at high-low=2.0


def test_atr_stop_piece_falls_back_when_not_enough_data():
    df = pd.DataFrame({"high": [102.0, 103.0], "low": [100.0, 101.0], "close": [101.0, 102.0]})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())
    setup = SetupResult(reference_level=100.0)

    stop = atr_stop(entry_price=110.0, setup=setup, ctx=ctx, params={"period": 14})

    assert stop == 110.0 * 0.995


def test_adx_strength_confirmation_piece_passes_on_strong_trend():
    values = np.arange(1, 61)
    df = pd.DataFrame({"high": values + 1, "low": values - 1, "close": values})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = adx_strength(ctx, {"period": 14, "min_adx": 25})

    assert result is not None
    assert result.extras["adx"] > 25


def test_adx_strength_confirmation_piece_none_on_choppy_range():
    n = 60
    close = 100 + np.sin(np.arange(n))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert adx_strength(ctx, {"period": 14, "min_adx": 25}) is None


def test_session_filter_piece_passes_for_an_allowed_session():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 17:00", tz="UTC")]})  # new_york hour (not the overlap)
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = session_filter(ctx, {"sessions": ["new_york", "london"]})

    assert result is not None
    assert result.extras["session"] == "new_york"


def test_session_filter_piece_none_for_a_disallowed_session():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 02:00", tz="UTC")]})  # asia hour
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert session_filter(ctx, {"sessions": ["new_york", "london"]}) is None


def test_session_filter_piece_requires_sessions_param():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 02:00", tz="UTC")]})
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    with pytest.raises(ValueError, match="requires a non-empty 'sessions' param"):
        session_filter(ctx, {})


def _synthetic_wandering_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    trend = np.linspace(0, 20, n)
    cycle = 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0, 0.5, n).cumsum() * 0.1
    close = 100 + trend + cycle + noise
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _known_bucket_values(table: pd.DataFrame) -> tuple[float, float]:
    """The (adx, rsi) midpoint of the table's most-populated bucket —
    guaranteed to be a bucket the table actually has data for, since
    per-dimension quantile bucketing doesn't guarantee every
    (adx_bucket, rsi_bucket) pair was jointly observed for an arbitrary
    real value."""
    biggest = table.sort_values("n_samples", ascending=False).iloc[0]
    adx_value = table.attrs["adx_edges"][int(biggest["adx_bucket"])].mid
    rsi_value = table.attrs["rsi_edges"][int(biggest["rsi_bucket"])].mid
    return float(adx_value), float(rsi_value)


def test_probability_trend_piece_uptrend_when_probability_clears_threshold(monkeypatch):
    from src.analysis.probability_table import build_frequency_table

    df = _synthetic_wandering_df()
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)
    adx_value, rsi_value = _known_bucket_values(table)

    monkeypatch.setattr("src.strategies.context.probability_trend.adx", lambda *_a, **_k: pd.Series([adx_value]))
    monkeypatch.setattr("src.strategies.context.probability_trend.rsi", lambda *_a, **_k: pd.Series([rsi_value]))

    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())
    trend = probability_trend(ctx, {"table": table, "min_probability": 0.0})

    assert trend == "uptrend"  # threshold of 0.0 always clears a known bucket


def test_probability_trend_piece_sideways_when_threshold_impossible(monkeypatch):
    from src.analysis.probability_table import build_frequency_table

    df = _synthetic_wandering_df()
    table = build_frequency_table(df, horizon_bars=5, n_buckets=5)
    adx_value, rsi_value = _known_bucket_values(table)

    monkeypatch.setattr("src.strategies.context.probability_trend.adx", lambda *_a, **_k: pd.Series([adx_value]))
    monkeypatch.setattr("src.strategies.context.probability_trend.rsi", lambda *_a, **_k: pd.Series([rsi_value]))

    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())
    trend = probability_trend(ctx, {"table": table, "min_probability": 1.1})

    assert trend == "sideways"


def test_probability_trend_piece_requires_a_table_param():
    ctx = EvalContext(price_window=_synthetic_wandering_df(), marked_window=pd.DataFrame())

    with pytest.raises(ValueError, match="requires a pre-built 'table' param"):
        probability_trend(ctx, {})


def test_scheduled_entry_piece_fires_at_the_configured_hour():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 03:00", tz="UTC")],
            "open": [100.0],
            "close": [101.0],
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    result = scheduled_entry(ctx, {"entry_hour": 3})

    assert result is not None
    assert result.reference_level == 101.0
    assert result.extras["scheduled_entry_hour"] == 3


def test_scheduled_entry_piece_none_outside_the_configured_hour():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 04:00", tz="UTC")],
            "open": [100.0],
            "close": [101.0],
        }
    )
    ctx = EvalContext(price_window=df, marked_window=pd.DataFrame())

    assert scheduled_entry(ctx, {"entry_hour": 3}) is None
