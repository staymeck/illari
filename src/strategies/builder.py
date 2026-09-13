"""Loads a YAML strategy config and resolves it against the catalog
registries into a ResolvedStrategy the backtest engine (src.backtest.engine)
can run directly.

See docs/PLAN.md, "Config-driven strategy catalog", and
config/strategies/trend_pullback_fib.yaml for the file shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

import src.strategies  # noqa: F401  (import triggers piece registration)
from src.strategies.registry import CONFIRMATIONS, CONTEXT, RISK_STOPS, RISK_TARGETS, SETUPS


@dataclass(frozen=True)
class ResolvedConfirmation:
    name: str
    fn: Callable
    params: dict


@dataclass(frozen=True)
class ResolvedStrategy:
    """Everything src.backtest.engine.run_backtest needs to simulate one
    strategy: resolved piece callables + their params, split into the same
    3 sections the user asked for — entry logic (context/setup/
    confirmations), risk (stop/target), and style (holding time/costs)."""

    name: str
    lookback_bars: int
    swing_order: int
    context_fn: Callable
    context_params: dict
    setup_fn: Callable
    setup_params: dict
    confirmations: list[ResolvedConfirmation]
    stop_fn: Callable
    stop_params: dict
    target_fn: Callable
    target_params: dict
    max_holding_bars: int
    fee_pct: float
    initial_equity: float


def load_strategy(path: str | Path) -> ResolvedStrategy:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    engine_cfg = raw.get("engine", {})
    context_cfg = raw["context"]
    setup_cfg = raw["setup"]
    risk_cfg = raw["risk"]
    style_cfg = raw.get("style", {})

    confirmations = [
        ResolvedConfirmation(
            name=c["piece"],
            fn=CONFIRMATIONS.get(c["piece"]),
            params=c.get("params", {}),
        )
        for c in raw.get("confirmations", [])
    ]

    return ResolvedStrategy(
        name=raw["name"],
        lookback_bars=engine_cfg.get("lookback_bars", 200),
        swing_order=engine_cfg.get("swing_order", 3),
        context_fn=CONTEXT.get(context_cfg["piece"]),
        context_params=context_cfg.get("params", {}),
        setup_fn=SETUPS.get(setup_cfg["piece"]),
        setup_params=setup_cfg.get("params", {}),
        confirmations=confirmations,
        stop_fn=RISK_STOPS.get(risk_cfg["stop"]["piece"]),
        stop_params=risk_cfg["stop"].get("params", {}),
        target_fn=RISK_TARGETS.get(risk_cfg["target"]["piece"]),
        target_params=risk_cfg["target"].get("params", {}),
        max_holding_bars=style_cfg.get("max_holding_bars", 24),
        fee_pct=style_cfg.get("fee_pct", 0.1),
        initial_equity=style_cfg.get("initial_equity", 10_000.0),
    )
