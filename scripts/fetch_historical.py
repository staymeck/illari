"""Pre-loads the local cache (data/raw/) with the 3 timeframes' candles for
every scenario defined in config/settings.yaml, without running the
backtest. Useful so you don't wait for the download during development.

Usage:
    python scripts/fetch_historical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_lab.cli import BIAS_BUFFER_DAYS, STRUCTURE_BUFFER_DAYS, _parse_ts, load_config  # noqa: E402
from trading_lab.data.cache import get_ohlcv  # noqa: E402
from datetime import timedelta  # noqa: E402


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "settings.yaml")
    exchange_id, pair, tf = cfg["exchange"], cfg["pair"], cfg["timeframes"]

    for scenario in cfg["scenarios"]:
        name = scenario["name"]
        start = _parse_ts(scenario["start"]).to_pydatetime()
        end = _parse_ts(scenario["end"]).to_pydatetime()

        print(f"[{name}] entry ({tf['entry']})...")
        get_ohlcv(exchange_id, pair, tf["entry"], start, end, f"{name}_entry")

        print(f"[{name}] structure ({tf['structure']})...")
        get_ohlcv(exchange_id, pair, tf["structure"], start - timedelta(days=STRUCTURE_BUFFER_DAYS), end, f"{name}_structure")

        print(f"[{name}] bias ({tf['bias']})...")
        get_ohlcv(exchange_id, pair, tf["bias"], start - timedelta(days=BIAS_BUFFER_DAYS), end, f"{name}_bias")

    print("Done. Data cached in data/raw/.")


if __name__ == "__main__":
    main()
