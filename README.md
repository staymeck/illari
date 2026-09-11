# Crypto trading analysis & backtesting lab (Phase 1)

A lab for testing, against historical crypto data, different **strategies**
(not just one) before risking real capital:

- **Multi-timeframe confluence**: daily bias + structure/Fibonacci on 1h +
  entry trigger on 5m (candle pattern + volume), with a least-squares
  support/resistance line (analytic geometry) that can either add an
  informational bonus or be required as a real filter.
- **Donchian Channel Breakout** ("Turtle Trading"): breakout of the recent
  high/low channel, aligned with the daily bias.

Each strategy supports several **variants** (different parameters), and all
of them run on the same historical scenarios for a side-by-side comparison
— including a **statistical significance test** that tells a real edge
apart from a lucky streak with few trades.

**This is only an analysis lab.** It doesn't execute real trades or give
financial advice — it simulates, on data that already happened, how well
(or poorly) each strategy would have done.

## 1. Set up the environment

From the project folder, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

(`pip install -e .` installs the `trading_lab` package in editable mode so
that `python -m trading_lab.cli` and the tests can find it.)

## 2. Configure strategies and scenarios

Edit [`config/settings.yaml`](config/settings.yaml):

- `scenarios`: the historical date ranges to compare (by default, 36
  months of bull trend in 2023-2025 and 36 months of range/crash in
  2019-2022, for BTC/USDT).
- `strategies`: a list, each with its `strategy_class` (`confluence` or
  `breakout` — see [`trading_lab/strategy/registry.py`](src/trading_lab/strategy/registry.py)),
  its `base_params`, and its `variants` (overrides on top of those base
  parameters). Each strategy's `base` variant is the one used for the
  candlestick chart and the report's probability table.

To add a new strategy: implement the `Strategy` interface
([`strategy/base.py`](src/trading_lab/strategy/base.py)) and register it in
`registry.py` — the rest of the lab (backtest, variant sweep, probability,
report) picks it up automatically.

## 3. Run the tests

```powershell
pytest
```

Covers: Fibonacci math, analytic geometry (least-squares lines), structure
detection (swings/trend) without lookahead, candle patterns, session
classification, multi-timeframe alignment (making sure it never looks at a
1h/1D candle that hasn't closed yet), the backtest engine (SL/TP execution,
commission/slippage, one position at a time), the statistical significance
test, and both strategies end-to-end with hand-designed synthetic data.

## 4. Run the backtest

```powershell
python -m trading_lab.cli backtest
```

Downloads (and caches in `data/raw/`, also indexed by date range — changing
a scenario's dates won't reuse the cache from a different range) the
candles needed from Binance, runs **every strategy × variant × scenario**
combination, and writes to `reports/`:

- `report.md`:
  - **Configuration comparison**: every strategy/variant × scenario
    combination, plus a ranking by the *worst* case across scenarios (so
    nothing overfit to a single period gets rewarded).
  - Per strategy (base variant): metrics, breakdown by UTC hour/session,
    **per-ingredient probability table** (with `p_value` and a
    significance flag — a large edge with few samples may not be
    significant), and the hit probability of the full signal.
- `reports/<scenario>/<strategy>/chart_5m.html` — interactive (Plotly)
  candlestick chart of that strategy on that scenario.
- `reports/<scenario>/<strategy>/equity.html` — equity curve.

To pre-load the data without running the backtest (useful during
development):

```powershell
python scripts/fetch_historical.py
```

## What this phase does NOT do (yet)

- It doesn't generate real-time signals or alerts — it's purely historical.
- It doesn't execute trades on any exchange.
- Entry hour/session is only an analytical breakdown in the report, not an
  active filter in the strategy.

These are the natural next steps, once this lab's results give confidence
in some strategy/configuration.

## Honest limitations

- Public OHLCV data doesn't carry real tick-by-tick order flow; the
  "volume" used is a reasonable proxy, not the exact figure.
- The backtest engine assumes that, when a candle touches both the stop
  loss and the take profit, the stop executes first (a conservative
  assumption).
- The significance test (two-proportion z-test) assumes a large sample;
  with few trades (tens or fewer) its results are unreliable — the report
  itself flags this via a small `n` and a high `p_value`.
- Past results don't guarantee future results — this lab's goal is to
  measure honestly, not to promise profitability.
