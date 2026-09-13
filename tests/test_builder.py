"""Tests for src/strategies/builder.py."""
import pytest

from src.strategies.builder import load_strategy

_MINIMAL_YAML = """
name: test_strategy
engine:
  lookback_bars: 50
  swing_order: 2
context:
  piece: dow_trend
  params: {lookback_swings: 3}
setup:
  piece: support_touch
  params: {tolerance_pct: 2.0}
confirmations:
  - piece: fibonacci
  - piece: volume
    params: {min_bias: 0.2}
risk:
  stop:
    piece: fixed_pct
    params: {pct_below_reference: 1.0}
  target:
    piece: risk_reward
    params: {ratio: 3.0}
style:
  max_holding_bars: 10
  fee_pct: 0.05
  initial_equity: 5000.0
"""

_UNKNOWN_PIECE_YAML = """
name: broken
context:
  piece: does_not_exist
setup:
  piece: support_touch
risk:
  stop: {piece: fixed_pct}
  target: {piece: risk_reward}
"""


def test_load_strategy_resolves_pieces_and_params(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(_MINIMAL_YAML, encoding="utf-8")

    strategy = load_strategy(path)

    assert strategy.name == "test_strategy"
    assert strategy.lookback_bars == 50
    assert strategy.swing_order == 2
    assert strategy.context_params == {"lookback_swings": 3}
    assert strategy.setup_params == {"tolerance_pct": 2.0}
    assert [c.name for c in strategy.confirmations] == ["fibonacci", "volume"]
    assert strategy.confirmations[1].params == {"min_bias": 0.2}
    assert strategy.stop_params == {"pct_below_reference": 1.0}
    assert strategy.target_params == {"ratio": 3.0}
    assert strategy.max_holding_bars == 10
    assert strategy.fee_pct == 0.05
    assert strategy.initial_equity == 5000.0


def test_load_strategy_defaults_when_engine_and_style_omitted(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(_UNKNOWN_PIECE_YAML.replace("does_not_exist", "dow_trend"), encoding="utf-8")

    strategy = load_strategy(path)

    assert strategy.lookback_bars == 200  # default
    assert strategy.swing_order == 3  # default
    assert strategy.max_holding_bars == 24  # default
    assert strategy.confirmations == []  # omitted entirely -> empty list


def test_load_strategy_unknown_piece_raises_clear_error(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(_UNKNOWN_PIECE_YAML, encoding="utf-8")

    with pytest.raises(KeyError, match="Unknown context piece 'does_not_exist'"):
        load_strategy(path)
