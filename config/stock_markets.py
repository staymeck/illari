"""US equity markets for the stock-market variant of the lab — same
"deliberately varied dynamics" principle as config/markets.py's 5 crypto
markets (a result must hold up across genuinely different assets, not
just one ticker's quirk wearing different names), spread across sectors
so a result isn't one industry's quirk either.

Expanded from an initial 5 to 20 after the first pass showed the strategy
is already so selective (by design, same as crypto) that 5 tickers only
produced 1-7 trades each — nowhere near enough to say anything (see
scripts/run_stock_baseline_backtest.py's own printed caveat). This widens
the sample without changing the strategy or the logic at all.

Sectors, deliberately spread (not weighted toward tech):
- Tech mega-cap: AAPL, MSFT, GOOGL
- Semiconductors (cyclical, AI-hype volatility): NVDA, AMD
- Consumer staples (macro-driven, not sentiment): PG, KO, WMT
- Growth/narrative-driven large cap: TSLA, AMZN
- Energy (oil-price-driven, a different macro factor): XOM, CVX
- Financials: JPM, BAC
- Healthcare: JNJ, UNH
- Industrials (cyclical, some idiosyncratic company risk): CAT, BA
- Sentiment/meme (the equity equivalent of DOGE — deliberate stress
  test, not expected to look good): GME, AMC

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
    StockMarket("AAPL", "tech mega-cap"),
    StockMarket("MSFT", "tech mega-cap — software/cloud"),
    StockMarket("GOOGL", "tech mega-cap — search/ads"),
    StockMarket("NVDA", "semiconductors — AI-hype narrative volatility"),
    StockMarket("AMD", "semiconductors — cyclical"),
    StockMarket("PG", "consumer staples — moves on macro/rates, not sentiment"),
    StockMarket("KO", "consumer staples"),
    StockMarket("WMT", "consumer staples / retail"),
    StockMarket("TSLA", "high-beta, narrative/retail-driven"),
    StockMarket("AMZN", "e-commerce/cloud growth large cap"),
    StockMarket("XOM", "cyclical, driven by oil prices"),
    StockMarket("CVX", "cyclical, driven by oil prices"),
    StockMarket("JPM", "financials/banking"),
    StockMarket("BAC", "financials/banking"),
    StockMarket("JNJ", "healthcare/pharma"),
    StockMarket("UNH", "healthcare/insurance"),
    StockMarket("CAT", "industrials, cyclical"),
    StockMarket("BA", "industrials, idiosyncratic company-specific risk"),
    StockMarket("GME", "sentiment/meme-driven — the equity equivalent of DOGE"),
    StockMarket("AMC", "sentiment/meme-driven"),
]

# Equities only trade during exchange sessions (see stock_fetcher.py's
# module docstring) — 1h/1d match the crypto lab's own timeframes exactly,
# for a direct, no-retuning comparison. 5m/30m intraday granularities are
# deliberately not offered yet: crypto's own history found finer timeframes
# dominated by noise relative to fixed costs (see config/markets.py), and
# there's no reason yet to expect equities to behave differently.
STOCK_TIMEFRAMES: list[str] = ["1h", "1d"]
