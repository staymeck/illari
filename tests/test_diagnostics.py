"""Tests for src/live/diagnostics.py — synthetic pieces, same convention
as test_paper_trading.py / test_engine_integration.py."""
from src.live.diagnostics import diagnose_entry, trace_entry_signal
from src.strategies.builder import ResolvedConfirmation, ResolvedStrategy
from src.strategies.risk.fixed_pct import fixed_pct_stop, risk_reward_target
from src.strategies.types import ConfirmationResult, EvalContext, SetupResult
from tests.test_engine_integration import _bar, _df

_LOOKBACK_BARS = 3


def _always_uptrend(ctx: EvalContext, params: dict) -> str:
    return "uptrend"


def _always_fires(ctx: EvalContext, params: dict) -> SetupResult | None:
    return SetupResult(reference_level=100.0)


def _never_fires(ctx: EvalContext, params: dict) -> SetupResult | None:
    return None


def _confirmation_that_passes(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    return ConfirmationResult(name="always_passes")


def _confirmation_that_fails(ctx: EvalContext, params: dict) -> ConfirmationResult | None:
    return None


def _strategy(setup_fn, confirmations: list[ResolvedConfirmation]) -> ResolvedStrategy:
    return ResolvedStrategy(
        name="test_strategy",
        lookback_bars=_LOOKBACK_BARS,
        swing_order=1,
        context_fn=_always_uptrend,
        context_params={},
        required_context="uptrend",
        setup_fn=setup_fn,
        setup_params={},
        confirmations=confirmations,
        stop_fn=fixed_pct_stop,
        stop_params={"pct_below_reference": 1.0},
        stop_trigger="intrabar",
        target_fn=risk_reward_target,
        target_params={"ratio": 2.0},
        max_holding_bars=10,
        fee_pct=0.1,
        initial_equity=10_000.0,
    )


def _ctx(df) -> EvalContext:
    return EvalContext(price_window=df, marked_window=df)


def test_trace_entry_signal_all_pass_gives_true_signal():
    strategy = _strategy(_always_fires, [ResolvedConfirmation(name="always_passes", fn=_confirmation_that_passes, params={})])
    df = _df([_bar(100, 100, 100, 100)] * 4)

    trace = trace_entry_signal(_ctx(df), strategy)

    assert trace["context_passed"] is True
    assert trace["setup_passed"] is True
    assert trace["confirmations"] == [{"piece": "always_passes", "passed": True}]
    assert trace["signal"] is True


def test_trace_entry_signal_does_not_short_circuit_on_failing_confirmation():
    # 2 confirmations: one fails first, one passes after - both must be
    # reported (unlike _find_entry_signal, which would stop at the first).
    confirmations = [
        ResolvedConfirmation(name="fails_first", fn=_confirmation_that_fails, params={}),
        ResolvedConfirmation(name="passes_second", fn=_confirmation_that_passes, params={}),
    ]
    strategy = _strategy(_always_fires, confirmations)
    df = _df([_bar(100, 100, 100, 100)] * 4)

    trace = trace_entry_signal(_ctx(df), strategy)

    assert trace["confirmations"] == [
        {"piece": "fails_first", "passed": False},
        {"piece": "passes_second", "passed": True},
    ]
    assert trace["signal"] is False


def test_trace_entry_signal_false_when_setup_never_fires():
    strategy = _strategy(_never_fires, [])
    df = _df([_bar(100, 100, 100, 100)] * 4)

    trace = trace_entry_signal(_ctx(df), strategy)

    assert trace["setup_passed"] is False
    assert trace["signal"] is False


def test_diagnose_entry_includes_timestamp_and_close_of_the_evaluated_bar():
    strategy = _strategy(_always_fires, [])
    df = _df([_bar(100, 100, 100, 100)] * 3 + [_bar(100, 105, 100, 103)])

    trace = diagnose_entry(df, None, strategy)

    assert trace["timestamp"] == str(df["timestamp"].iloc[3])
    assert trace["close"] == 103.0
    assert trace["signal"] is True
