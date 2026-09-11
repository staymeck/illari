"""Maps the short name used in `config/settings.yaml` (`strategy_class`) to
the concrete class — so `sweep.py`/`cli.py` don't have to import each
strategy by hand or use `if/elif` chains scattered across the code."""

from __future__ import annotations

from trading_lab.strategy.base import Strategy
from trading_lab.strategy.breakout_strategy import DonchianBreakoutStrategy
from trading_lab.strategy.confluence_strategy import ConfluenceStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "confluence": ConfluenceStrategy,
    "breakout": DonchianBreakoutStrategy,
}


def get_strategy_class(strategy_class: str) -> type[Strategy]:
    try:
        return STRATEGY_REGISTRY[strategy_class]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy_class: '{strategy_class}' (known: {known})") from exc
