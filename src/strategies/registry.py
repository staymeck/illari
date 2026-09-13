"""Generic name -> callable registry, one instance per catalog layer.

See docs/PLAN.md, "Config-driven strategy catalog", for the 4-layer design
(context, setup, confirmations, risk) this backs. Piece modules under
src/strategies/{context,setups,confirmations,risk}/*.py import the relevant
registry below and register into it with `@REGISTRY.register("name")`; each
layer's package `__init__.py` imports every piece module so registration
happens as a side effect of `import src.strategies`.
"""
from __future__ import annotations

from typing import Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._pieces: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            if name in self._pieces:
                raise ValueError(f"{self.kind} piece '{name}' is already registered")
            self._pieces[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        try:
            return self._pieces[name]
        except KeyError:
            available = ", ".join(sorted(self._pieces)) or "(none registered)"
            raise KeyError(f"Unknown {self.kind} piece '{name}'. Available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._pieces)


CONTEXT = Registry("context")
SETUPS = Registry("setup")
CONFIRMATIONS = Registry("confirmation")
RISK_STOPS = Registry("risk stop")
RISK_TARGETS = Registry("risk target")
