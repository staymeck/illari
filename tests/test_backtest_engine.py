import pandas as pd
import pytest

from trading_lab.backtest.engine import run_backtest
from trading_lab.strategy.base import Signal


def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=5 * i)


def _entry_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = [_ts(i) for i in range(len(rows))]
    return df


def _signal(direction="long", entry_price=100.0, stop_loss=95.0, take_profit=110.0, ts_idx=0) -> Signal:
    return Signal(
        timestamp=_ts(ts_idx),
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence_score=4,
        hour_utc=0,
        session="asia",
        reasons={},
    )


def test_take_profit_hit_no_costs():
    df = _entry_df(
        [
            {"open": 99, "high": 101, "low": 98, "close": 100},   # signal, enters at the close
            {"open": 100, "high": 103, "low": 99, "close": 102},  # touches neither SL nor TP
            {"open": 102, "high": 111, "low": 101, "close": 109}, # touches TP (110)
            {"open": 109, "high": 112, "low": 108, "close": 110},
        ]
    )
    sig = _signal()

    result = run_backtest([sig], df, initial_capital=10000, risk_per_trade_pct=1.0,
                           commission_pct=0.0, slippage_pct=0.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(110.0)
    # risk_amount = 10000*1% = 100; risk_per_unit = 100-95 = 5; position_size = 20
    assert trade.position_size == pytest.approx(20.0)
    # pnl = (110-100)*20 = 200
    assert trade.pnl == pytest.approx(200.0)
    assert result.final_equity == pytest.approx(10200.0)


def test_stop_loss_hit_before_take_profit_in_same_bar_is_conservative():
    df = _entry_df(
        [
            {"open": 99, "high": 101, "low": 98, "close": 100},
            # a candle that touches BOTH the SL (95) and the TP (110): SL must be assumed first.
            {"open": 100, "high": 111, "low": 94, "close": 105},
        ]
    )
    sig = _signal()

    result = run_backtest([sig], df, initial_capital=10000, risk_per_trade_pct=1.0,
                           commission_pct=0.0, slippage_pct=0.0)

    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(95.0)
    # pnl = (95-100)*20 = -100
    assert trade.pnl == pytest.approx(-100.0)


def test_only_one_open_position_at_a_time():
    df = _entry_df(
        [
            {"open": 99, "high": 101, "low": 98, "close": 100},
            {"open": 100, "high": 103, "low": 99, "close": 102},
            {"open": 102, "high": 111, "low": 101, "close": 109},
            {"open": 109, "high": 112, "low": 108, "close": 110},
        ]
    )
    sig1 = _signal(ts_idx=0)
    sig2 = _signal(ts_idx=1)  # generated while sig1 is still open -> should be ignored

    result = run_backtest([sig1, sig2], df, initial_capital=10000, risk_per_trade_pct=1.0,
                           commission_pct=0.0, slippage_pct=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].entry_time == _ts(0)


def test_commission_and_slippage_reduce_pnl():
    df = _entry_df(
        [
            {"open": 99, "high": 101, "low": 98, "close": 100},
            {"open": 100, "high": 103, "low": 99, "close": 102},
            {"open": 102, "high": 111, "low": 101, "close": 109},
        ]
    )
    sig = _signal()

    no_cost = run_backtest([sig], df, initial_capital=10000, risk_per_trade_pct=1.0,
                            commission_pct=0.0, slippage_pct=0.0)
    with_cost = run_backtest([sig], df, initial_capital=10000, risk_per_trade_pct=1.0,
                              commission_pct=0.1, slippage_pct=0.05)

    assert with_cost.trades[0].pnl < no_cost.trades[0].pnl
    # Entry should get worse (more expensive) for a long with slippage.
    assert with_cost.trades[0].entry_price > no_cost.trades[0].entry_price
    # Exit should get worse (cheaper) for a long with slippage.
    assert with_cost.trades[0].exit_price < no_cost.trades[0].exit_price


def test_risk_multiplier_scales_position_size():
    df = _entry_df(
        [
            {"open": 99, "high": 101, "low": 98, "close": 100},
            {"open": 100, "high": 103, "low": 99, "close": 102},
            {"open": 102, "high": 111, "low": 101, "close": 109},
        ]
    )
    full_risk = _signal()  # risk_multiplier defaults to 1.0
    half_risk = Signal(
        timestamp=_ts(0), direction="long", entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        confidence_score=4, hour_utc=0, session="asia", reasons={}, risk_multiplier=0.5,
    )

    result_full = run_backtest([full_risk], df, initial_capital=10000, risk_per_trade_pct=1.0,
                                commission_pct=0.0, slippage_pct=0.0)
    result_half = run_backtest([half_risk], df, initial_capital=10000, risk_per_trade_pct=1.0,
                                commission_pct=0.0, slippage_pct=0.0)

    # risk_amount = 10000*1%*0.5 = 50; risk_per_unit = 5 -> position_size = 10 (half of the full-risk case).
    assert result_half.trades[0].position_size == pytest.approx(result_full.trades[0].position_size * 0.5)
    assert result_half.trades[0].pnl == pytest.approx(result_full.trades[0].pnl * 0.5)


def test_equity_curve_starts_at_initial_capital():
    df = _entry_df([{"open": 99, "high": 101, "low": 98, "close": 100}])
    result = run_backtest([], df, initial_capital=5000, risk_per_trade_pct=1.0,
                           commission_pct=0.0, slippage_pct=0.0)
    assert result.equity_curve["equity"].iloc[0] == pytest.approx(5000.0)
    assert result.final_equity == pytest.approx(5000.0)
