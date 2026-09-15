"""Common interface for strategies and the structure of a generated signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    direction: str  # 'long' | 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence_score: int
    hour_utc: int
    session: str
    reasons: dict = field(default_factory=dict)
    # Multiplies `risk_per_trade_pct` at execution time (see backtest/engine.py).
    # Default 1.0 = no change from existing fixed-fraction sizing. Strategies
    # only set this away from 1.0 when `dynamic_risk_enabled` scales it by
    # the current volatility regime (see indicators/volatility.py).
    risk_multiplier: float = 1.0


class Strategy(ABC):
    """Every strategy receives the three timeframe DataFrames (already with
    their own indicators computed) and returns a list of signals, each
    with a confidence score instead of being a binary decision."""

    @abstractmethod
    def generate_signals(
        self,
        entry_df: pd.DataFrame,
        structure_df: pd.DataFrame,
        bias_df: pd.DataFrame,
        params: dict,
    ) -> list[Signal]:
        raise NotImplementedError
