"""Tests for src/backtest/parallel.py.

`_fake_fetch` is a plain module-level function (not a lambda or closure) so
it can be pickled and sent to worker processes — this keeps the test fully
offline, with no real network access or real backtest computation needed."""
import pandas as pd

from src.backtest.parallel import BacktestJob, run_jobs_in_parallel


def _fake_fetch(symbol: str, timeframe: str, since, until) -> pd.DataFrame:
    # Flat, uneventful price series: runs the engine end-to-end without
    # producing any trades, which keeps the test fast and deterministic.
    idx = pd.date_range("2024-01-01", periods=250, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
        }
    )


def test_run_jobs_in_parallel_runs_all_jobs_and_returns_metrics():
    jobs = [
        BacktestJob("BTC/USDT", "1h", "2024-01-01", "2024-01-11", scenario="a"),
        BacktestJob("ETH/USDT", "1h", "2024-01-01", "2024-01-11", scenario="b"),
        BacktestJob("SOL/USDT", "1h", "2024-01-01", "2024-01-11", scenario="c"),
    ]

    results = run_jobs_in_parallel(jobs, fetch_fn=_fake_fetch, max_workers=3)

    assert len(results) == 3
    symbols = {r.job.symbol for r in results}
    assert symbols == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    for r in results:
        assert r.metrics["n_trades"] == 0  # flat price series: no confluence, no trades
