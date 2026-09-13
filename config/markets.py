"""Markets, timeframes, and trading-session configuration for the lab.

Reference: docs/PLAN.md — "Phase 1 scope".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    symbol: str  # Binance spot symbol, ccxt format (e.g. "BTC/USDT")
    role: str    # why it's in the set (documentation only, not used in calculations)


# The 5 markets agreed in docs/PLAN.md. SOL/USDT chosen over BNB/USDT as the
# higher-beta altcoin; run scripts/check_listing_dates.py to confirm the real
# listing date of each one before fixing scenario windows.
MARKETS: list[Market] = [
    Market("BTC/USDT", "base reference, blue-chip crypto"),
    Market("ETH/USDT", "blue chip with its own dynamics (DeFi/smart contracts)"),
    Market("PAXG/USDT", "tokenized gold — the only genuinely non-crypto asset in the set"),
    Market("SOL/USDT", "higher-beta/volatility altcoin"),
    Market("DOGE/USDT", "driven by sentiment/social media rather than fundamentals"),
]

# Multi-context timeframes: fine-grained (entries), intermediate, and global.
TIMEFRAMES: list[str] = ["5m", "1h", "1d"]

# Market sessions in UTC (start/end hours, end exclusive). Used to label each
# simulated trade and break down results by session.
SESSIONS_UTC: dict[str, tuple[int, int]] = {
    "asia": (0, 8),
    "london": (7, 9),              # London open, overlap resolved below
    "overlap_london_ny": (12, 16),
    "new_york": (13, 21),
    "off_hours": (21, 24),
}


def session_for_hour(hour_utc: int) -> str:
    """Returns the market session (UTC) for a given hour, prioritizing the
    London-NY overlap when applicable, same precedence as the previous report
    (resources/report.md)."""
    if 12 <= hour_utc < 16:
        return "overlap_london_ny"
    if 13 <= hour_utc < 21:
        return "new_york"
    if 7 <= hour_utc < 12:
        return "london"
    if 0 <= hour_utc < 8:
        return "asia"
    return "off_hours"
