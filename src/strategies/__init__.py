"""Config-driven strategy catalog: context/setup/confirmation/risk pieces,
resolved from a YAML file by src.strategies.builder.

See docs/PLAN.md, "Config-driven strategy catalog".
"""
from src.strategies import confirmations, context, risk, setups  # noqa: F401  (registers every piece)
