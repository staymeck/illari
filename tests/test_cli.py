import pandas as pd

from trading_lab.cli import resolve_scenario_window

SCENARIO = {"name": "mixed_volatility_2019_2022", "start": "2019-07-01T00:00:00Z", "end": "2022-06-30T00:00:00Z"}


def test_resolve_scenario_window_uses_scenario_dates_without_override():
    market = {"name": "BTC", "pair": "BTC/USDT"}

    start, end = resolve_scenario_window(SCENARIO, market)

    assert start == pd.Timestamp("2019-07-01T00:00:00Z")
    assert end == pd.Timestamp("2022-06-30T00:00:00Z")


def test_resolve_scenario_window_applies_market_override():
    # PAXG-style case: listed after the scenario's default start.
    market = {
        "name": "PAXG",
        "pair": "PAXG/USDT",
        "scenario_overrides": {"mixed_volatility_2019_2022": {"start": "2020-08-28T00:00:00Z"}},
    }

    start, end = resolve_scenario_window(SCENARIO, market)

    assert start == pd.Timestamp("2020-08-28T00:00:00Z")
    assert end == pd.Timestamp("2022-06-30T00:00:00Z")  # end wasn't overridden -> keeps the scenario's default


def test_resolve_scenario_window_ignores_override_for_other_scenarios():
    other_scenario = {"name": "bull_trend_2023_2025", "start": "2023-01-01T00:00:00Z", "end": "2025-12-31T00:00:00Z"}
    market = {
        "name": "PAXG",
        "pair": "PAXG/USDT",
        "scenario_overrides": {"mixed_volatility_2019_2022": {"start": "2020-08-28T00:00:00Z"}},
    }

    start, end = resolve_scenario_window(other_scenario, market)

    assert start == pd.Timestamp("2023-01-01T00:00:00Z")
