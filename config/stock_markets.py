"""US equity markets for the stock-market variant of the lab — the same
"deliberately varied dynamics" principle as config/markets.py's 5 crypto
markets (a result must hold up across genuinely different assets, not
just one ticker's quirk wearing different names), mapped onto stocks:

- AAPL: mega-cap tech blue chip — the most liquid single stock in the
  world, plays the role BTC plays among crypto markets.
- PG (Procter & Gamble): consumer staples, moves on rates/macro rather
  than sentiment — the equity equivalent of PAXG's "non-crypto-sentiment"
  role.
- TSLA: high-beta, narrative/retail-driven — large swings on news,
  sentiment-heavy but still has real fundamentals underneath, sits
  between SOL and DOGE in character.
- XOM (ExxonMobil): cyclical, driven by oil prices — a genuinely
  different macro driver than tech sentiment or consumer spending.
- GME (GameStop): the closest equity equivalent of DOGE — driven by
  social-media/retail sentiment more than fundamentals. Deliberately
  included as a stress test: if the strategy's edge is real it should
  struggle here same as everywhere else; if it only "works" on GME's
  narrative-driven runs, that's a red flag, not a result to celebrate.

Data comes from Alpaca (src/data/stock_fetcher.py), not Binance — this
file is otherwise structurally identical to config/markets.py so the same
scripts/engine can consume either with a one-line source swap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StockMarket:
    symbol: str  # Alpaca ticker (e.g. "AAPL")
    role: str    # why it's in the set (documentation only, not used in calculations)


STOCK_MARKETS: list[StockMarket] = [
    StockMarket("AAPL", "mega-cap tech blue chip, base reference"),
    StockMarket("PG", "consumer staples — moves on macro/rates, not sentiment"),
    StockMarket("TSLA", "high-beta, narrative/retail-driven"),
    StockMarket("XOM", "cyclical, driven by oil prices — a different macro driver"),
    StockMarket("GME", "sentiment/meme-driven — the equity equivalent of DOGE"),
]

# Equities only trade during exchange sessions (see stock_fetcher.py's
# module docstring) — 1h/1d match the crypto lab's own timeframes exactly,
# for a direct, no-retuning comparison. 5m/30m intraday granularities are
# deliberately not offered yet: crypto's own history found finer timeframes
# dominated by noise relative to fixed costs (see config/markets.py), and
# there's no reason yet to expect equities to behave differently.
STOCK_TIMEFRAMES: list[str] = ["1h", "1d"]
