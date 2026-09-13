# Illari — Quantitative trading & analysis lab (Phase 1)

## Context

This project (`illari`) was already underway in a previous session: there was
an agreed plan, a Binance data fetcher, backtests running, and an initial
report (`resources/report.md`, with only 2 sample scenarios of very small
size: 18 and 7 trades). The PC got reformatted and all of that work was lost —
the repository wasn't even under `git init`. All that survived are the two
PDFs and the old report in `resources/`.

This plan reconstructs, in writing and version-controlled in git, everything
that had been agreed in the prior conversation, so it doesn't depend on the
memory of a chat session. The goal of this first phase **was not to code
yet** — it was to document the full lab design so it could be confirmed
before writing any code. (That plan was approved, and the first implementation
step is now in progress — see `CLAUDE.md`/commit history for current status.)

**Done in that session:** `git init`, `main` branch, initial commit with
`resources/`.

## Project objective

Build a **backtesting lab** that evaluates, with objective and measurable
rules (not an AI black box), how well a trading strategy would have performed
based on:

- Market structure (support/resistance, higher/lower highs and lows, breaks of structure)
- Analytic geometry applied to candles (trendlines, channels, Fibonacci
  retracements as confluence zones — not as a standalone signal)
- Candlestick patterns combined with context (never isolated patterns)
- Buying/selling pressure (volume, and funding rate as a proxy for market bias)
- The hour/session of entry into the market, as an analysis variable

Explicit philosophy (already agreed): the goal is not to "predict" the market
deterministically. The goal is to measure probabilities with objective rules
and rigorous backtesting — most bots fail by skipping the rigor of this phase,
not because of the strategy itself.

### The project's 3 phases (full vision, current focus on Phase 1)

1. **Historical lab (focus of this plan):** run the strategy over past data
   across several markets/scenarios and measure results (return, drawdown,
   win rate, profit factor, breakdown by hour/session).
2. **Live signals (future):** the same analysis engine running on live data,
   producing a "snapshot" of the market's current condition and its
   associated historical probability — without executing anything yet.
3. **Automated execution (future, conditional):** only if Phases 1 and 2 show
   consistent results, connect to an exchange to trade autonomously.

## Phase 1 scope

### Markets (all on Binance, via the same API — no new infrastructure)

| Symbol | Role |
|---|---|
| BTC/USDT | Base reference, blue-chip crypto |
| ETH/USDT | Blue chip with its own dynamics (DeFi/smart contracts) |
| PAXG/USDT | Tokenized gold — the only genuinely non-crypto asset in the set, no new data source needed |
| SOL/USDT or BNB/USDT | Higher-beta/volatility altcoin |
| DOGE/USDT | Driven by sentiment/social media, not fundamentals — genuinely different dynamics |

**Implementation caveat:** not all of them have traded on Binance since the
same date (e.g. SOL was listed after BTC/ETH). Before fixing scenario dates,
each symbol's real listing date must be checked against the API, and the
window adjusted per asset instead of forcing the same range for all 5.

### Time contexts (multi-timeframe, global market view)

The same analysis runs across three timeframes so the global view isn't lost:

- **5 minutes** — fine-grained entries/timing
- **1 hour** — intermediate trend context
- **1 day** — global trend context (is the underlying market bullish, bearish, or sideways?)

### Entry hour as an analysis variable (not execution, yet)

Every simulated trade is tagged with its UTC hour and market session (Asia,
London, New York, London-NY overlap, off-hours), and performance is reported
broken down by hour/session — just like the old report already did
(`resources/report.md`, "Breakdown by entry hour" and "by session" sections).
In other words: **analysis only for now**, with **UTC** as the reference time
zone. If any hourly pattern turns out to be consistent and significant, it
will be evaluated later as a possible entry filter.

### Technical analysis engine components

1. **Market structure:** objective code rules (not "eyeballed") for
   higher/lower highs and lows, breaks of structure, support/resistance zones.
2. **Fibonacci:** used as a confluence zone (an interest filter), never as a
   standalone signal.
3. **Candlestick patterns:** only considered combined with structural context
   (where they appear), never as a signal on their own.
4. **Buying/selling pressure:** per-candle volume as a direct proxy; if
   Binance volume isn't enough, funding rate (see data sources below) is used
   as a second proxy for buyer/seller bias.
5. Reference material already available in
   `resources/12_claves_analisis_tecnico.pdf` (Dow Theory, support/resistance,
   price patterns, Elliott, oscillators, ADX, money management) — used as a
   source of concrete rules when implementing each component.
   (`resources/Mercado_analisi.pdf` is a generic market-study/marketing
   textbook excerpt, not relevant to this technical part — archived unused.)

### Additional data sources (all free, already evaluated)

- **Funding rate** (Binance Futures, via `ccxt` — the same library already
  used for candles): updates every 8h, full history available.
- **Fear & Greed Index** (`alternative.me`, free API): one value per day.
- *Deferred for now:* on-chain flows and granular social sentiment — these
  require paid services (Glassnode/CryptoQuant/Santiment) or a home-grown
  wallet-labeling project. Left for a possible later stage if the rest shows
  a real edge.

### Statistical tools (an upgrade over comparing loose groups)

- **Autocorrelation:** does today's return predict tomorrow's, across several lags at once?
- **Regression:** fit future return = f(past returns, volume, volatility) and
  measure real R², instead of a binary verdict.
- **Granger causality:** does funding rate or the Fear & Greed Index have
  formal predictive power over price, beyond simple correlation?

Autocorrelation and regression can be applied already with the candle data
being downloaded; Granger causality applies once funding rate/Fear & Greed are integrated.

### Backtesting rigor (to avoid repeating the previous attempt's problems)

- Walk-forward / out-of-sample validation (never fit and test on the same segment).
- Real Binance commission and slippage costs included in every simulation.
- Sufficient sample sizes before drawing conclusions — the previous report had
  scenarios of 18 and 7 trades, not enough to conclude anything; new scenarios
  should aim for a reasonable minimum trade count (to be defined with real
  data, but clearly more than a few dozen).
- Report broken down by scenario x market x timeframe x hour/session, with the
  same metrics already used (win rate, profit factor, expectancy, max
  drawdown, total return).

## Tech stack

- **Language:** Python.
- **Data:** Binance public API (`ccxt`) for candles + funding rate;
  `alternative.me` for Fear & Greed. No credentials needed in this phase
  (public data reads only).
- No order execution in this phase — Phase 1 is 100% read-only and simulation.

## Proposed project structure

```
illari/
├── resources/              # (already exists) reference PDFs + old report
├── data/                   # downloaded candles/funding/F&G, cached on disk
├── src/
│   ├── data/                 # fetcher.py (Binance/ccxt), funding.py, fear_greed.py
│   ├── analysis/              # structure.py, fibonacci.py, candles.py, volume.py
│   ├── stats/                  # autocorrelation.py, regression.py, granger.py
│   ├── backtest/                # simulation engine, walk-forward, metrics
│   └── report/                  # report.md generation + interactive charts
├── config/                  # symbols, scenario dates, timeframes, UTC sessions
├── tests/
└── requirements.txt
```

This is a reasonable starting point, adjustable during actual implementation.

## Immediate next steps (once this plan is approved)

1. Create `requirements.txt` and the folder structure (`src/`, `data/`, `config/`).
2. Implement `src/data/fetcher.py` (multi-timeframe OHLCV candles via
   `ccxt`/Binance) and validate the real listing date of each of the 5 symbols.
3. Implement `src/analysis/structure.py` (support/resistance, trend) as the
   first objective, independently testable component.
4. Minimal end-to-end backtest on BTC/USDT on 1 timeframe, to validate the
   full engine (data -> signal -> simulation -> report) before scaling up to
   5 markets x 3 timeframes.
5. Only then add Fibonacci, candlesticks, volume, funding rate, Fear & Greed,
   and the statistical tools.

## Verification

- Every module in `src/analysis` and `src/stats` gets unit tests on synthetic
  data (cases where the expected result is known in advance).
- The minimal backtest (step 4) must run end-to-end and produce a
  `report.md` with metrics + hour/session breakdown, just like the old
  report but regenerated from code version-controlled in this repo.
- Before accepting any strategy as good: check sample size and out-of-sample
  performance, not just the in-sample result.

## Strategy catalog (config-driven, added after the initial scale-up)

Once the 5-market x 4-timeframe scan showed the one hardcoded strategy losing
money everywhere, the engine was generalized into a **catalog** of
interchangeable pieces instead of one fixed set of rules, so new strategies
can be assembled and backtested by writing a YAML file, not new Python.

Four layers, each a named registry under `src/strategies/`:

- **context** (`src/strategies/context/`) — is the market trending/ranging?
  `dow_trend` (swing-point based) and `ma_trend` (moving-average cross).
- **setup** (`src/strategies/setups/`) — where to look for an entry:
  `support_touch`, `breakout` (Donchian-style), `mean_reversion`
  (lower-Bollinger-Band touch).
- **confirmations** (`src/strategies/confirmations/`, a list — all must
  pass) — `fibonacci`, `candlestick`, `volume`, `rsi_momentum`,
  `macd_momentum`, `vwap_bias`, `adx_strength` (trend *strength*, not just
  direction), `session_filter` (the original "hora de entrada" idea, now an
  actual filter).
- **risk** (`src/strategies/risk/`) — `fixed_pct` or `atr_multiple` stop +
  `risk_reward` target. A stop's `trigger` (`intrabar`, default, or `close`)
  controls when it confirms — see "Kaufman validation" below.

`src/strategies/builder.py` resolves a YAML file (see
`config/strategies/*.yaml`) into a `ResolvedStrategy`; `src/backtest/engine.py`
runs it generically (context -> setup -> confirmations -> risk), long-only,
with the required market regime itself configurable via `context.required`
(default `uptrend`; `mean_reversion_bb.yaml` sets `any` since gating a
mean-reversion setup behind an uptrend would defeat the point).

`config/strategies/trend_pullback_fib.yaml` reproduces the original
hardcoded strategy exactly (verified: all 20 five-market x four-timeframe
results identical before/after the refactor). `breakout_momentum.yaml` and
`mean_reversion_bb.yaml` are two more catalog strategies proving the pieces
actually compose into different, runnable strategies.

Explicitly out of scope for this catalog (don't fit the same entry-signal
pipeline): grid trading, arbitrage/pairs/stat-arb, fundamental/event/news
trading, machine learning.

## Validated against Kaufman, *Trading Systems and Methods* (5th ed.)

`resources/TSaM.pdf` — a serious, widely-respected reference (Wiley Trading),
well beyond the two intro-level PDFs the lab started with. Checked our
approach against it directly rather than assuming; concretely:

- **Multi-market testing = robustness, not redundancy.** "If a technique
  works in the Swiss franc but not the euro... the method is most likely not
  robust but fine-tuned to each market" (ch. 21) — this is exactly why the
  lab runs every strategy across all 5 markets instead of trusting one.
- **Dropping 5m is the expected, documented outcome, not a quirk of our
  setup.** Kaufman's own 17-market test across 2-80 day calculation periods:
  "faster trends are uniformly losses, while progressively longer trends are
  profitable" — fast periods "generate too many trades with small profits
  and losses that will not be greater than the cost." Matches our 5m result
  (390-524 trades, -46% to -66%, every strategy variant tried) closely
  enough that 5m was dropped from `config/markets.py` entirely.
- **ADX as a trend-strength filter, at nearly our own threshold.** Ch. 23's
  market-ranking framework uses "Wilder's ADX... but only for values greater
  than 0.20" — we'd already picked `min_adx: 25` for `adx_strength`
  independently; close enough to be reassuring rather than a coincidence to
  chase.
- **Stops are "a duel with price noise"** (ch. 23) — a stop can capture the
  worst exit right before a recovery. Kaufman's own fix: "stop-loss orders
  are usually based on the closing price, or... the actual exit is still on
  the close" — implemented as `risk.stop.trigger: close` (see
  `config/strategies/trend_pullback_close_stop.yaml`), alongside a new
  `stop_was_premature` diagnostic per trade (src/backtest/engine.py) and a
  "Stop-loss noise diagnostic" report section (src/report/render.py)
  measuring exactly what fraction of stop-outs would have reached target
  anyway. Targets deliberately do NOT get the same close-confirmation —
  Kaufman recommends the opposite for profit-taking (capture the intraday
  spike immediately, don't wait for the close).
- **Fixed-% stops are the least favored option** — "the ones most likely to
  work must adapt to volatility... rather than a fixed dollar amount or a
  percentage of price" — consistent with adding `atr_multiple` as a second
  stop piece, though our own test (BTC/USDT, same ATR multiple on 5m vs 1h)
  found the same caution Kaufman gives for his own ATR-stop test: it's a
  single-market/period result, and the right multiple needs its own tuning
  per timeframe, not one setting that transfers everywhere.

**Gaps this surfaced that the catalog still doesn't cover** (candidates for
later, not built yet): trailing stops ("more practical than initial stops,"
per Kaufman) is a real hole — the catalog only has initial stops; multiple/
partial profit targets (scaling out in stages instead of one target);
volatility-based position sizing as an alternative to a stop rather than a
companion to it; and a "% of profitable tests across a parameter range"
robustness metric (ch. 21) — the lab currently reports one point estimate
per strategy run, not a sweep, which is a real rigor gap relative to
Kaufman's own testing standard.
